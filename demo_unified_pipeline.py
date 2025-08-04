import torch
from PIL import Image
import numpy as np
from typing import List

# Diffusers và Transformers imports
from diffusers.utils import load_image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from huggingface_hub import hf_hub_download

# Imports từ các tệp trong dự án
# Đảm bảo các tệp này nằm trong cùng thư mục hoặc trong PYTHONPATH
from unified_story_inpaint_pipeline import UnifiedStoryInpaintPipeline
from FLUX_Controlnet_Inpainting.controlnet_flux import FluxControlNetModel
from FLUX_Controlnet_Inpainting.transformer_flux import FluxTransformer2DModel
from anystory.module import AnyStoryReduxImageEncoder

# ==========================================================================================
# 1. Cấu hình
# ==========================================================================================

# --- Đường dẫn đến các ảnh đầu vào ---
# Cung cấp 2 đến 5 ảnh tham chiếu để duy trì tính nhất quán của nhân vật
REF_IMAGE_PATHS = [
    'https://huggingface.co/datasets/junjie96/AnyStory/resolve/main/characters/Saitama.png',
    # Thêm ảnh tham chiếu khác ở đây nếu muốn, ví dụ:
    # 'path/to/your/second/ref_image.png', 
]

# Ảnh chính và mask cho vùng cần inpainting
CONTROL_IMAGE_PATH = 'https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha/resolve/main/images/bucket.png'
CONTROL_MASK_PATH = 'https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha/resolve/main/images/bucket_mask.jpeg'

# --- Prompt và các tham số ---
PROMPT = 'a man with a bald head wearing a white shoe, carrying a white bucket with text "FLUX" on it'
NEGATIVE_PROMPT = "ugly, deformed, disfigured, poor details, bad anatomy"
OUTPUT_PATH = "unified_result.png"

# --- Đường dẫn đến các model checkpoint ---
BASE_FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
FLUX_REDUX_MODEL_ID = "black-forest-labs/FLUX.1-Redux-dev"
CONTROLNET_INPAINT_MODEL_ID = "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha"
ANYSTORY_MODEL_REPO = "Junjie96/AnyStory"
ANYSTORY_MODEL_FILENAME = "anystory_flux.bin"
SEGMENTATION_MODEL_ID = "ZhengPeng7/BiRefNet"

# ==========================================================================================
# Helper Function: Tự động tạo Mask cho ảnh tham chiếu (từ storyboard.py)
# ==========================================================================================

def get_ref_mask_and_crop(pil_image: Image.Image, segmentation_model, device: str) -> (Image.Image, Image.Image):
    """
    Sử dụng mô hình BiRefNet để tách nền và crop ảnh tham chiếu.
    Trả về ảnh đã được che nền và mask của nó.
    """
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
    if len(non_zero_indices[0]) == 0: # Nếu mask trống
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
# 2. Tải các mô hình và Pipeline
# ==========================================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16

    print("Loading models...")
    # Tải ControlNet
    controlnet = FluxControlNetModel.from_pretrained(CONTROLNET_INPAINT_MODEL_ID, torch_dtype=torch_dtype)

    # Tải Transformer gốc của FLUX
    transformer = FluxTransformer2DModel.from_pretrained(
        BASE_FLUX_MODEL_ID, subfolder='transformer', torch_dtype=torch_dtype
    )
    
    # Tải các thành phần phụ trợ của FLUX (VAE, Text Encoders,...)
    # và khởi tạo Unified Pipeline
    pipe = UnifiedStoryInpaintPipeline.from_pretrained(
        BASE_FLUX_MODEL_ID,
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )
    
    # Tải các thành phần của AnyStory
    anystory_path = hf_hub_download(repo_id=ANYSTORY_MODEL_REPO, filename=ANYSTORY_MODEL_FILENAME)
    
    # Tải checkpoint của AnyStory vào các thành phần tương ứng trong pipeline
    print("Loading AnyStory weights...")
    anystory_state_dict = torch.load(anystory_path, weights_only=True, map_location="cpu")
    
    # LoRA cho Ref
    pipe.load_lora_weights(anystory_state_dict["ref"], adapter_name="ref_lora")
    
    # Redux Embedder
    pipe.redux_embedder.load_state_dict(anystory_state_dict["redux"], strict=False)
    
    # Router Embedder and Adapter
    router_embedder_sd = {}
    router_adapter_sd = {}
    router_lora_sd = {}
    for k, v in anystory_state_dict["router"].items():
        if k.startswith("router_embedder."):
            router_embedder_sd[k.replace("router_embedder.", "")] = v
        elif "attn.processor" in k:
            router_adapter_sd[k.replace("transformer.", "")] = v
        else:
            router_lora_sd[k] = v
            
    pipe.router_embedder.load_state_dict(router_embedder_sd, strict=False)
    pipe.transformer.load_state_dict(router_adapter_sd, strict=False)
    pipe.load_lora_weights(router_lora_sd, adapter_name="router_lora")
    
    # Deactivate LoRA adapters by default
    pipe.transformer.set_adapters(["ref_lora", "router_lora"], [0.0, 0.0])
    
    pipe.to(device)

    # Tải mô hình segmentation
    print("Loading segmentation model...")
    segmentation_model = AutoModelForImageSegmentation.from_pretrained(SEGMENTATION_MODEL_ID, trust_remote_code=True)
    segmentation_model.to(device).eval()

    print("Models loaded successfully.")

    # ==========================================================================================
    # 3. Chuẩn bị ảnh đầu vào
    # ==========================================================================================
    print("Preparing input images...")
    # Tải ảnh và mask cho ControlNet Inpainting
    output_size = (768, 768)
    control_image = load_image(CONTROL_IMAGE_PATH).convert("RGB").resize(output_size)
    control_mask = load_image(CONTROL_MASK_PATH).convert("RGB").resize(output_size)

    # Tải và xử lý ảnh tham chiếu cho AnyStory
    ref_images_processed: List[Image.Image] = []
    ref_masks_processed: List[Image.Image] = []
    
    for path in REF_IMAGE_PATHS:
        ref_image = load_image(path)
        print(f"Processing reference image: {path}")
        masked_ref_image, ref_mask = get_ref_mask_and_crop(ref_image, segmentation_model, device)
        ref_images_processed.append(masked_ref_image)
        ref_masks_processed.append(ref_mask)
        
    print(f"Processed {len(ref_images_processed)} reference images.")

    # ==========================================================================================
    # 4. Chạy pipeline
    # ==========================================================================================
    generator = torch.Generator(device=device).manual_seed(42)

    print("Running unified pipeline...")
    result_image = pipe(
        # --- Prompt ---
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        
        # --- Kích thước ảnh ---
        height=output_size[1],
        width=output_size[0],
        
        # --- Tham số của ControlNet Inpainting ---
        control_image=control_image,
        control_mask=control_mask,
        controlnet_conditioning_scale=0.9,
        
        # --- Tham số của AnyStory ---
        ref_images=ref_images_processed,
        ref_masks=ref_masks_processed,
        redux_scale=0.8, # Điều chỉnh cường độ của AnyStory
        
        # --- Tham số chung ---
        num_inference_steps=28,
        guidance_scale=3.5,
        true_guidance_scale=1.0,
        generator=generator,
    ).images[0]

    # ==========================================================================================
    # 5. Lưu kết quả
    # ==========================================================================================
    result_image.save(OUTPUT_PATH)
    print(f"Successfully generated and saved image to {OUTPUT_PATH}")