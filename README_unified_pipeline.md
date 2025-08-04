# Unified Story Inpaint Pipeline

## Tổng quan

`UnifiedStoryInpaintPipeline` là một pipeline thống nhất kết hợp hai kỹ thuật mạnh mẽ:

1. **AnyStory**: Tạo hình ảnh với nhân vật nhất quán dựa trên hình ảnh tham chiếu
2. **Flux ControlNet Inpainting**: Tô màu lại (inpainting) các vùng cụ thể với kiểm soát không gian

Pipeline này cho phép bạn:
- Tạo storyboard với nhân vật nhất quán
- Chỉnh sửa hình ảnh có kiểm soát
- Kết hợp cả hai khả năng cho các tác vụ phức tạp

## Cài đặt

```bash
pip install diffusers==0.30.2
pip install transformers
pip install torch torchvision
```

## Cách sử dụng

### 1. Khởi tạo Pipeline

```python
import torch
from unified_story_inpaint_pipeline import UnifiedStoryInpaintPipeline
from diffusers import FluxPipeline, FluxControlNetModel
from transformers import SiglipImageProcessor, SiglipVisionModel

# Load các mô hình cơ sở
flux_pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha", torch_dtype=torch.bfloat16)

# Load AnyStory components
redux_embedder = AnyStoryReduxImageEncoder.from_pretrained("path/to/anystory/redux")
router_embedder = AnyStoryReduxImageEncoder.from_pretrained("path/to/anystory/router")

# Load SigLIP components cho AnyStory
siglip_processor = SiglipImageProcessor(size={"height": 384, "width": 384})
siglip_encoder = SiglipVisionModel.from_pretrained("path/to/siglip")

# Khởi tạo unified pipeline
pipeline = UnifiedStoryInpaintPipeline(
    scheduler=flux_pipeline.scheduler,
    vae=flux_pipeline.vae,
    text_encoder=flux_pipeline.text_encoder,
    tokenizer=flux_pipeline.tokenizer,
    text_encoder_2=flux_pipeline.text_encoder_2,
    tokenizer_2=flux_pipeline.tokenizer_2,
    transformer=flux_pipeline.transformer,
    controlnet=controlnet,
    redux_embedder=redux_embedder,
    router_embedder=router_embedder,
    siglip_image_encoder=siglip_encoder,
    siglip_image_processor=siglip_processor,
).to("cuda")
```

### 2. Sử dụng chỉ AnyStory (Character Consistency)

```python
from PIL import Image

# Hình ảnh tham chiếu cho nhân vật
ref_images = [Image.open("character1.jpg"), Image.open("character2.jpg")]
ref_masks = [Image.open("character1_mask.jpg"), Image.open("character2_mask.jpg")]

# Tạo hình ảnh với nhân vật nhất quán
result = pipeline(
    prompt="Two characters sitting in a cafe, having a conversation",
    ref_images=ref_images,
    ref_masks=ref_masks,
    ref_start_at=0.0,
    ref_end_at=1.0,
    num_inference_steps=28,
    guidance_scale=3.5,
    height=1024,
    width=1024,
)

result.images[0].save("anystory_output.png")
```

### 3. Sử dụng chỉ ControlNet Inpainting

```python
# Hình ảnh gốc và mask
control_image = Image.open("original_image.jpg")
control_mask = Image.open("inpaint_mask.jpg")

# Tô màu lại vùng được chỉ định
result = pipeline(
    prompt="A beautiful landscape with mountains in the background",
    control_image=control_image,
    control_mask=control_mask,
    controlnet_conditioning_scale=0.9,
    num_inference_steps=28,
    guidance_scale=3.5,
    height=1024,
    width=1024,
)

result.images[0].save("inpaint_output.png")
```

### 4. Kết hợp cả hai kỹ thuật

```python
# Tạo hình ảnh với nhân vật nhất quán và inpainting
result = pipeline(
    prompt="A character walking in a beautiful garden",
    ref_images=[Image.open("character.jpg")],
    ref_masks=[Image.open("character_mask.jpg")],
    control_image=Image.open("garden_image.jpg"),
    control_mask=Image.open("garden_mask.jpg"),
    ref_start_at=0.0,
    ref_end_at=1.0,
    controlnet_conditioning_scale=0.9,
    num_inference_steps=28,
    guidance_scale=3.5,
    height=1024,
    width=1024,
)

result.images[0].save("combined_output.png")
```

## Tham số chính

### AnyStory Parameters
- `ref_images`: Danh sách hình ảnh tham chiếu cho nhân vật
- `ref_masks`: Danh sách mask cho hình ảnh tham chiếu
- `ref_start_at`: Thời điểm bắt đầu áp dụng reference conditioning (0.0-1.0)
- `ref_end_at`: Thời điểm kết thúc reference conditioning (0.0-1.0)
- `enable_ref_cache`: Bật/tắt cache cho reference conditioning
- `ref_shift`: Độ dịch chuyển vị trí reference
- `enable_ref_mask`: Bật/tắt mask cho reference

### ControlNet Parameters
- `control_image`: Hình ảnh điều khiển cho inpainting
- `control_mask`: Mask xác định vùng cần inpainting
- `controlnet_conditioning_scale`: Mức độ ảnh hưởng của ControlNet (0.9-1.0)

### General Parameters
- `prompt`: Prompt văn bản cho generation
- `negative_prompt`: Negative prompt
- `num_inference_steps`: Số bước denoising (28-50)
- `guidance_scale`: Guidance scale cho CFG (3.5-7.0)
- `height/width`: Kích thước hình ảnh đầu ra
- `generator`: Generator cho reproducibility

## Lưu ý quan trọng

1. **Memory Efficiency**: Pipeline chỉ load các mô hình cơ sở một lần và tái sử dụng cho cả hai kỹ thuật.

2. **Resolution**: Tối ưu cho 1024x1024, có thể sử dụng các kích thước khác nhưng có thể ảnh hưởng đến chất lượng.

3. **ControlNet Scale**: Khuyến nghị sử dụng 0.9-1.0 cho `controlnet_conditioning_scale`.

4. **Reference Timing**: Điều chỉnh `ref_start_at` và `ref_end_at` để kiểm soát khi nào reference conditioning được áp dụng.

5. **Multiple References**: Có thể sử dụng nhiều hình ảnh tham chiếu cho cùng một nhân vật hoặc nhiều nhân vật khác nhau.

## Ví dụ nâng cao

### Storyboard Generation với Multiple Characters

```python
# Tạo storyboard với nhiều nhân vật
characters = [
    {"image": "hero.jpg", "mask": "hero_mask.jpg", "name": "Hero"},
    {"image": "villain.jpg", "mask": "villain_mask.jpg", "name": "Villain"},
    {"image": "sidekick.jpg", "mask": "sidekick_mask.jpg", "name": "Sidekick"}
]

ref_images = [Image.open(char["image"]) for char in characters]
ref_masks = [Image.open(char["mask"]) for char in characters]

scenes = [
    "Hero and Sidekick planning their next move",
    "Villain plotting in his lair",
    "Hero confronting Villain in epic battle"
]

for i, scene in enumerate(scenes):
    result = pipeline(
        prompt=scene,
        ref_images=ref_images,
        ref_masks=ref_masks,
        num_inference_steps=28,
        guidance_scale=3.5,
        height=1024,
        width=1024,
    )
    result.images[0].save(f"storyboard_scene_{i+1}.png")
```

### Inpainting với Character Consistency

```python
# Inpainting một vùng cụ thể trong khi giữ nhân vật nhất quán
result = pipeline(
    prompt="Character wearing a red dress in a modern office",
    ref_images=[Image.open("character.jpg")],
    ref_masks=[Image.open("character_mask.jpg")],
    control_image=Image.open("office_with_hole.jpg"),
    control_mask=Image.open("hole_mask.jpg"),
    ref_start_at=0.0,
    ref_end_at=1.0,
    controlnet_conditioning_scale=0.9,
    num_inference_steps=28,
    guidance_scale=3.5,
)

result.images[0].save("office_inpaint_with_character.png")
```

## Troubleshooting

1. **Out of Memory**: Giảm `num_inference_steps` hoặc sử dụng `torch.cuda.empty_cache()`

2. **Poor Character Consistency**: Tăng `ref_start_at` và `ref_end_at` range

3. **Weak Inpainting Control**: Tăng `controlnet_conditioning_scale`

4. **Slow Generation**: Sử dụng ít inference steps hơn hoặc giảm resolution

## License

Pipeline này tuân theo license của các mô hình cơ sở:
- FLUX: Non-Commercial License
- AnyStory: Theo license của Alibaba
- ControlNet: Theo license của Alimama Creative 