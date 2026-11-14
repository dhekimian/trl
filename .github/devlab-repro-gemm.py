"""Standalone repro attempts for a GPU memory fault seen on gfx1201.

In trl's TestAsyncGRPOTrainerVLM::test_train_vlm_text_only_data (tiny Qwen3.5-MoE, bf16), the HIP runtime
reports "Memory Fault Error ... kernel: Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x32_MI16x16x1_..."
(a hipBLASLt bf16 GEMM) during `model.generate`, with torch 2.15.0+rocm7.14 and +rocm10.0.

Modes (each prints the case before running it, so the last line before a crash names the culprit):
  gemm        F.linear in bf16, with and without bias, over the model's shapes and neighbours.
  generate    the model alone: model.generate as the test's server calls it, 30 times.
  concurrent  generate in one thread while another thread trains a second copy, as in the test.
"""

import faulthandler
import sys
import threading

import torch
import torch.nn.functional as F

faulthandler.enable()
MODEL = "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6"
dev = "cuda"


def say(*a):
    print(*a, flush=True)


def gemm():
    # hidden 16, head_dim 256, 4 q heads / 2 kv heads, MoE/shared intermediate 32, linear-attn key/value heads
    # 16x128 / 32x128, vocab 248320; plus powers of two around them.
    dims = sorted({16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 248320})
    ms = list(range(1, 18)) + [24, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 256, 512]
    n = 0
    for bias in (True, False):
        for k in dims:
            for nout in dims:
                if k * nout > 248320 * 1024:
                    continue
                w = torch.randn(nout, k, device=dev, dtype=torch.bfloat16)
                b = torch.randn(nout, device=dev, dtype=torch.bfloat16) if bias else None
                for m in ms:
                    say(f"case m={m} k={k} n={nout} bias={bias}")
                    x = torch.randn(1, m, k, device=dev, dtype=torch.bfloat16)
                    y = F.linear(x, w, b)
                    torch.cuda.synchronize()
                    n += 1
                del w, b
    say(f"gemm: {n} cases, no fault")


def load():
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)


def gen_once(model, i, g):
    plen = int(torch.randint(8, 64, (1,), generator=g))
    ids = torch.randint(0, 1000, (1, plen), generator=g).to(dev)
    out = model.generate(
        ids, attention_mask=torch.ones_like(ids), do_sample=True, max_new_tokens=8, temperature=1.0,
        top_p=1.0, top_k=0, output_scores=True, return_dict_in_generate=True,
    )
    torch.cat(out.scores).float().log_softmax(-1)
    torch.cuda.synchronize()


def generate():
    model = load()
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for i in range(30):
            say(f"generate {i}")
            gen_once(model, i, g)
    say("generate: 30 calls, no fault")


def concurrent():
    server, trainee = load(), load().float()
    opt = torch.optim.SGD(trainee.parameters(), lr=0.1)
    err = []

    def serve():
        g = torch.Generator().manual_seed(0)
        try:
            with torch.no_grad():
                for i in range(30):
                    say(f"concurrent generate {i}")
                    gen_once(server, i, g)
        except Exception as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=serve)
    t.start()
    step = 0
    while t.is_alive() and not err:
        ids = torch.randint(0, 1000, (3, 40), device=dev)
        loss = trainee(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        step += 1
    t.join()
    if err:
        raise err[0]
    say(f"concurrent: 30 generate calls alongside {step} training steps, no fault")


if __name__ == "__main__":
    say(torch.__version__, torch.version.hip, torch.cuda.get_device_properties(0).gcnArchName)
    {"gemm": gemm, "generate": generate, "concurrent": concurrent}[sys.argv[1]]()
