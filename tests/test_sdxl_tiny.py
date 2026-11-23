"""Wire-up test for the real SDXL backend code, using a *tiny random* SDXL + ControlNet +
IP-Adapter built in memory (same recipe as diffusers' own tests). It proves the calls we
make (masked multi-image IP-Adapter, per-image scales, ControlNet on/off, dummy image when
there is no reference) are accepted by diffusers, on CPU, in a few seconds. Image quality is
meaningless here; real weights are used on Kaggle."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from PIL import Image  # noqa: E402

from egig.backends import GenerationJob  # noqa: E402
from egig.backends.sdxl import SDXLBackend  # noqa: E402
from egig.config import Settings  # noqa: E402

pytestmark = pytest.mark.slow


def tiny_pipeline():
    from diffusers import (AutoencoderKL, ControlNetModel, EulerDiscreteScheduler,
                           StableDiffusionXLControlNetPipeline, UNet2DConditionModel)
    from transformers import (CLIPImageProcessor, CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection,
                              CLIPTokenizer, CLIPVisionConfig, CLIPVisionModelWithProjection)

    torch.manual_seed(0)
    sdxl_like = dict(block_out_channels=(32, 64), layers_per_block=2, in_channels=4,
                     down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), attention_head_dim=(2, 4),
                     use_linear_projection=True, addition_embed_type="text_time", addition_time_embed_dim=8,
                     transformer_layers_per_block=(1, 2), projection_class_embeddings_input_dim=80,
                     cross_attention_dim=64)
    unet = UNet2DConditionModel(sample_size=32, out_channels=4, up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                                **sdxl_like)
    controlnet = ControlNetModel(conditioning_embedding_out_channels=(16, 32), **sdxl_like)
    vae = AutoencoderKL(block_out_channels=[32, 64], in_channels=3, out_channels=3, latent_channels=4,
                        down_block_types=["DownEncoderBlock2D"] * 2, up_block_types=["UpDecoderBlock2D"] * 2)
    text_cfg = CLIPTextConfig(bos_token_id=0, eos_token_id=2, hidden_size=32, intermediate_size=37,
                              layer_norm_eps=1e-05, num_attention_heads=4, num_hidden_layers=5, pad_token_id=1,
                              vocab_size=1000, hidden_act="gelu", projection_dim=32)
    tok = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
    image_encoder = CLIPVisionModelWithProjection(CLIPVisionConfig(
        hidden_size=32, projection_dim=32, num_hidden_layers=2, num_attention_heads=4, image_size=32,
        intermediate_size=37, patch_size=4))
    pipe = StableDiffusionXLControlNetPipeline(
        vae=vae, text_encoder=CLIPTextModel(text_cfg), text_encoder_2=CLIPTextModelWithProjection(text_cfg),
        tokenizer=tok, tokenizer_2=tok, unet=unet, controlnet=controlnet,
        scheduler=EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, steps_offset=1,
                                         beta_schedule="scaled_linear", timestep_spacing="leading"),
        image_encoder=image_encoder, feature_extractor=CLIPImageProcessor(crop_size=32, size=32),
    )

    # IP-Adapter weights in the original checkpoint layout: image projection + to_k_ip/to_v_ip per cross-attention
    cross = unet.config.cross_attention_dim
    image_proj = {"proj.weight": torch.randn(4 * cross, 32) * 0.02, "proj.bias": torch.zeros(4 * cross),
                  "norm.weight": torch.ones(cross), "norm.bias": torch.zeros(cross)}
    ip_layers, key_id = {}, 1
    for name in unet.attn_processors:
        if name.endswith("attn1.processor"):
            continue
        if name.startswith("mid_block"):
            hidden = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            hidden = list(reversed(unet.config.block_out_channels))[int(name[len("up_blocks.")])]
        else:
            hidden = unet.config.block_out_channels[int(name[len("down_blocks.")])]
        ip_layers[f"{key_id}.to_k_ip.weight"] = torch.randn(hidden, cross) * 0.02
        ip_layers[f"{key_id}.to_v_ip.weight"] = torch.randn(hidden, cross) * 0.02
        key_id += 2
    unet._load_ip_adapter_weights([{"image_proj": image_proj, "ip_adapter": ip_layers}])
    return pipe


@pytest.fixture(scope="module")
def backend():
    return SDXLBackend(Settings(), pipe=tiny_pipeline())


def masks(size):
    left, right = Image.new("L", (size, size), 0), Image.new("L", (size, size), 0)
    left.paste(255, (0, 0, size // 2, size))
    right.paste(255, (size // 2, 0, size, size))
    return left, right


def job(**kw):
    base = dict(prompt="a dog in front of a temple", negative_prompt="blurry", width=64, height=64, steps=2,
                guidance=5.0, seed=3)
    base.update(kw)
    return GenerationJob(**base)


def test_two_references_with_masks_scales_and_controlnet(backend):
    refs = [Image.new("RGB", (80, 60), (200, 50, 50)), Image.new("RGB", (60, 90), (50, 50, 200))]
    edges = Image.new("RGB", (64, 64))
    edges.paste((255, 255, 255), (10, 10, 54, 12))
    out = backend.generate(job(ip_images=refs, ip_masks=list(masks(64)), ip_scales=[0.6, 0.4],
                               control_image=edges, control_scale=0.5, control_end=0.6))
    assert out.size == (64, 64)
    procs = [p for p in backend.pipe.unet.attn_processors.values() if hasattr(p, "scale")]
    assert procs and procs[0].scale == [[0.6, 0.4]]  # one adapter, one scale per reference image


def test_no_reference_and_no_controlnet(backend):
    out = backend.generate(job())  # dummy grey IP image at scale 0, black control image at scale 0
    assert out.size == (64, 64)


def test_single_reference(backend):
    out = backend.generate(job(ip_images=[Image.new("RGB", (64, 64), "green")], ip_masks=[masks(64)[0]],
                               ip_scales=[0.7]))
    assert out.size == (64, 64)


def test_token_count_uses_the_pipeline_tokenizer(backend):
    assert 0 < backend.count_tokens("a photo of a dog") <= 10
