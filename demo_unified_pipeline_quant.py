import torch
from PIL import Image
import numpy as np
from typing import List
import gc

# Diffusers và Transformers imports
from diffusers.utils import load_image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from huggingface_hub import hf_hub_download
from transformers.models.siglip import SiglipImageProcessor, SiglipVisionModel

# Thêm import cho quantization
try:
    from optimum.quanto import quantize, freeze, qint8
    OPTIMUM_AVAILABLE = True
except ImportError:
    print("Optimum library is not installed. Skipping quantization. To turn on this mode, run: pip install optimum[quanto]")
    OPTIMUM_AVAILABLE = False

# Imports từ các tệp trong dự án
from unified_story_inpaint_pipeline import UnifiedStoryInpaintPipeline
from FLUX_Controlnet_Inpainting.controlnet_flux import FluxControlNetModel
from FLUX_Controlnet_Inpainting.transformer_flux import FluxTransformer2DModel
from anystory.module import AnyStoryReduxImageEncoder

# ==========================================================================================
# Cấu hình (Không đổi)
# ==========================================================================================
QUANTIZE_MODEL = True
ENABLE_CPU_OFFLOAD = False
PROMPT = 'A character is sitting on a chair, full body visible, front facing, clearly showing his face.'
NEGATIVE_PROMPT = "ugly, deformed, disfigured, poor details, bad anatomy"
OUTPUT_PATH = "unified_result_quantized.png"
BASE_FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
FLUX_REDUX_MODEL_ID = "black-forest-labs/FLUX.1-Redux-dev"
CONTROLNET_INPAINT_MODEL_ID = "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha"
ANYSTORY_MODEL_REPO = "Junjie96/AnyStory"
ANYSTORY_MODEL_FILENAME = "anystory_flux.bin"
SEGMENTATION_MODEL_ID = "ZhengPeng7/BiRefNet"

# ==========================================================================================
# Helper Function
# ==========================================================================================
def get_ref_mask_and_crop(pil_image: Image.Image, segmentation_model, device: str) -> (Image.Image, Image.Image):
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_images = transform_image(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = segmentation_model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(pil_image.size)
    mask = ((np.array(mask) > 200) * 255).astype(np.uint8)
    non_zero_indices = np.nonzero(mask)
    if len(non_zero_indices[0]) == 0:
        return pil_image, Image.fromarray(np.ones_like(mask) * 255)
    min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
    s = max(max_x - min_x, max_y - min_y) * 1.04
    min_x, max_x = int(cx - s // 2), int(cx + s // 2)
    min_y, max_y = int(cy - s // 2), int(cy + s // 2)
    pil_image_cropped = pil_image.convert("RGB").crop((min_x, min_y, max_x, max_y))
    pil_mask_cropped = Image.fromarray(mask).crop((min_x, min_y, max_x, max_y))
    np_image = np.array(pil_image_cropped)
    np_mask = np.array(pil_mask_cropped)[..., None] / 255.
    pil_masked_image = Image.fromarray((np_mask * np_image + (1 - np_mask) * 255.).astype(np.uint8))
    return pil_masked_image, pil_mask_cropped

# ==========================================================================================
# KỊCH BẢN CHÍNH
# ==========================================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16

    # ---Create MASK ---
    print("--- Part 1: Pre-processing Reference Masks ---")
    print("Loading segmentation model...")
    segmentation_model = AutoModelForImageSegmentation.from_pretrained(SEGMENTATION_MODEL_ID, trust_remote_code=True)
    segmentation_model.to(device).eval()

    ref_images_processed, ref_masks_processed = [], []
    saitama_local_path = "assets/examples/thayTriet.png"
    ref_image = load_image(saitama_local_path)
    print(f"Processing reference image: {saitama_local_path}")
    masked_ref_image, ref_mask = get_ref_mask_and_crop(ref_image, segmentation_model, device)
    ref_images_processed.append(masked_ref_image)
    ref_masks_processed.append(ref_mask)
    print(f"Processed {len(ref_images_processed)} reference images.")

    for i in range(len(ref_images_processed)):
        ref_images_processed[i].save(f"ref_images_processed_{i}.png")
        ref_masks_processed[i].save(f"ref_masks_processed_{i}.png")

    # --- GIẢI PHÓNG VRAM ---
    print("Freeing VRAM by moving segmentation model to CPU...")
    del segmentation_model
    gc.collect()
    torch.cuda.empty_cache()

    # --- Loading and Optimizing Main Pipeline ---
    print("\n--- Part 2: Loading and Optimizing Main Pipeline ---")
    # Load all components to CPU
    print("Loading all model components to CPU first...")
    controlnet = FluxControlNetModel.from_pretrained(CONTROLNET_INPAINT_MODEL_ID, torch_dtype=torch_dtype)
    transformer = FluxTransformer2DModel.from_pretrained(BASE_FLUX_MODEL_ID, subfolder='transformer', torch_dtype=torch_dtype)
    redux_embedder = AnyStoryReduxImageEncoder.from_pretrained(FLUX_REDUX_MODEL_ID, subfolder="image_embedder", torch_dtype=torch_dtype)
    router_embedder = AnyStoryReduxImageEncoder.from_pretrained(FLUX_REDUX_MODEL_ID, subfolder="image_embedder", torch_dtype=torch_dtype)
    siglip_image_processor = SiglipImageProcessor(size={"height": 384, "width": 384})
    siglip_image_encoder = SiglipVisionModel.from_pretrained(FLUX_REDUX_MODEL_ID, subfolder="image_encoder", torch_dtype=torch_dtype)
    
    # Initializing the complete pipeline on CPU
    print("Initializing the complete pipeline on CPU...")
    pipe = UnifiedStoryInpaintPipeline.from_pretrained(
        BASE_FLUX_MODEL_ID, controlnet=controlnet, transformer=transformer, redux_embedder=redux_embedder,
        router_embedder=router_embedder, siglip_image_encoder=siglip_image_encoder,
        siglip_image_processor=siglip_image_processor, torch_dtype=torch_dtype,
    )
    
    # Loading AnyStory weights
    print("Loading AnyStory weights...")
    anystory_path = hf_hub_download(repo_id=ANYSTORY_MODEL_REPO, filename=ANYSTORY_MODEL_FILENAME)
    anystory_state_dict = torch.load(anystory_path, weights_only=True, map_location="cpu")
    pipe.load_lora_weights(anystory_state_dict["ref"], adapter_name="ref_lora")
    pipe.redux_embedder.load_state_dict(anystory_state_dict["redux"], strict=False, assign=True)
    router_embedder_sd, router_adapter_sd, router_lora_sd = {}, {}, {}
    for k, v in anystory_state_dict["router"].items():
        if k.startswith("router_embedder."): router_embedder_sd[k.replace("router_embedder.", "")] = v
        elif "attn.processor" in k: router_adapter_sd[k.replace("transformer.", "")] = v
        else: router_lora_sd[k] = v
    pipe.router_embedder.load_state_dict(router_embedder_sd, strict=False, assign=True)
    pipe.transformer.load_state_dict(router_adapter_sd, strict=False)
    pipe.load_lora_weights(router_lora_sd, adapter_name="router_lora")
    pipe.transformer.set_adapters(["ref_lora", "router_lora"], [0.0, 0.0])

    # Starting quantization on CPU
    if QUANTIZE_MODEL and OPTIMUM_AVAILABLE:
        print("Starting quantization on CPU...")
        
        # Quantizing core models
        print("Quantizing core models...")
        quantize(pipe.transformer, weights=qint8); freeze(pipe.transformer)
        # quantize(pipe.text_encoder, weights=qint8); freeze(pipe.text_encoder)
        quantize(pipe.text_encoder_2, weights=qint8); freeze(pipe.text_encoder_2)
        quantize(pipe.controlnet, weights=qint8); freeze(pipe.controlnet)
        
        # # Quantizing VAE
        # print("Quantizing VAE...")
        # quantize(pipe.vae, weights=qint8); freeze(pipe.vae)
        
        # # Quantizing SigLIP image encoder
        # print("Quantizing SigLIP image encoder...")
        # quantize(pipe.siglip_image_encoder, weights=qint8); freeze(pipe.siglip_image_encoder)
        
        # # AnyStory Embedders
        # print("Quantizing AnyStory embedders...")
        # quantize(pipe.redux_embedder, weights=qint8); freeze(pipe.redux_embedder)
        # quantize(pipe.router_embedder, weights=qint8); freeze(pipe.router_embedder)
        
        print("Quantization on CPU complete.")

    # Moving the optimized pipeline to GPU
    print("Moving the optimized pipeline to GPU...")
    pipe.to(device)

    # Enabling CPU Offloading
    if ENABLE_CPU_OFFLOAD:
        print("Enabling CPU Offloading...")
        pipe.enable_model_cpu_offload()

    print("All models loaded and optimized successfully.")
    torch.cuda.empty_cache()

    # --- Preparing Inpainting Images and Running Pipeline ---
    print("\n--- Part 3: Preparing Inpainting Images and Running Pipeline ---")
    output_size = (512, 512)
    control_image_path_local = "assets/examples/control_image.png"
    control_mask_path_local = "assets/examples/control_mask.png"
    control_image = load_image(control_image_path_local).convert("RGB").resize(output_size)
    control_mask = load_image(control_mask_path_local).convert("RGB").resize(output_size)
    # control_image = None
    # control_mask = None
    generator = torch.Generator(device=device).manual_seed(2025)
    print("Running unified pipeline...")
    result_image = pipe(
        prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT, height=output_size[1], width=output_size[0],
        control_image=control_image, control_mask=control_mask, controlnet_conditioning_scale=0.5,
        ref_images=ref_images_processed, ref_masks=ref_masks_processed,
        num_inference_steps=30, guidance_scale=3.5, true_guidance_scale=3.5, generator=generator,
    ).images[0]

    # --- PHẦN 4: Save the results ---
    result_image.save(OUTPUT_PATH)
    print(f"Successfully generated and saved image to {OUTPUT_PATH}")
