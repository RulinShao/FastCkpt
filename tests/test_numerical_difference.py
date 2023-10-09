import time
import torch

def test_ckpt(test_hf_grad_ckpt=False, sequence_length=1024, batch_size=1, repeat=True):
    # Need to call this before importing transformers.
    if test_hf_grad_ckpt:
        from fastckpt.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
    else:
        from fastckpt.llama_flash_attn_ckpt_monkey_patch import replace_hf_ckpt_with_fast_ckpt, clear_all_buffers_at_the_end_of_training
        replace_hf_ckpt_with_fast_ckpt()

    import transformers

    model_name_or_path = "Llama-2-7b-chat-hf"
    model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            ).half().cuda()
    

    model.model.gradient_checkpointing = True
    model.train()
    warmup_steps = 20
    total_steps = 60

    torch.manual_seed(42)
    inputs_embeds = torch.randn((batch_size, sequence_length, 4096), requires_grad=True).half().cuda()
    torch.cuda.synchronize()
    time_per_iter = []
    for i in range(total_steps):
        start = time.time()
        outputs = model(inputs_embeds=inputs_embeds, use_cache=False)
        loss = torch.mean(outputs.logits)
        loss.backward()
        torch.cuda.synchronize()
        end = time.time()
        if i >= warmup_steps:
            time_per_iter.append(end-start)
        if not test_hf_grad_ckpt:
            clear_all_buffers_at_the_end_of_training()
        if not repeat:
            break

    avg_time = sum(time_per_iter) / len(time_per_iter) if repeat else end-start
    print(f"Avg forward + backward spent {avg_time}s.")
    out = [outputs.logits.detach()]
    out += [model.model.layers[1].mlp.up_proj.weight.grad]
    out += [model.model.layers[1].self_attn.q_proj.weight.grad]
    out += [model.model.layers[0].mlp.up_proj.weight.grad]
    out += [model.model.layers[0].self_attn.q_proj.weight.grad]
    return out


def test_numerical_difference():
    my_ckpt_out = test_ckpt(False, repeat=False)
    hf_ckpt_out = test_ckpt(True, repeat=False)
    for i, (my_out, hf_out) in enumerate(zip(my_ckpt_out, hf_ckpt_out)):
        assert torch.allclose(my_out, hf_out)
        print(f"Passed {i}-th check!")


if __name__ == '__main__':
    test_numerical_difference()