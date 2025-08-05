# Copyright (c) 2024 Unified Story Inpaint Pipeline
# This script merges AnyStory character consistency with Flux ControlNet Inpainting

import gc
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    is_torch_version,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

# Import custom components from AnyStory
from anystory.attention_processor import AnyStoryFluxAttnProcessor2_0
from anystory.module import AnyStoryReduxImageEncoder
from anystory.transformer import tranformer_forward
from anystory.block import block_forward, single_block_forward
from anystory.lora_controller import enable_lora

# Import ControlNet components
from FLUX_Controlnet_Inpainting.controlnet_flux import FluxControlNetModel
from FLUX_Controlnet_Inpainting.transformer_flux import FluxTransformer2DModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

# Utility functions from Flux pipeline
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    """Calculate shift parameter for positional encoding."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Retrieve timesteps from scheduler."""
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

# Step 1: Define the Merged Transformer Logic
def merged_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    ref_hidden_states: torch.FloatTensor = None,
    ref_temb: torch.FloatTensor = None,
    ref_rotary_emb=None,
    router_hidden_states: torch.FloatTensor = None,
    router_temb: torch.FloatTensor = None,
    router_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
    controlnet_block_sample: Optional[torch.Tensor] = None,
):
    """
    Merged block forward function that combines AnyStory and ControlNet logic.
    
    This function:
    1. Executes the full AnyStory block logic first
    2. Adds the ControlNet signal as a residual at the end
    """
    use_ref_cond = ref_hidden_states is not None
    use_router = router_hidden_states is not None
    
    # Execute AnyStory block logic first
    encoder_hidden_states, hidden_states, ref_hidden_states, router_hidden_states = block_forward(
        self,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
        ref_hidden_states=ref_hidden_states,
        ref_temb=ref_temb,
        ref_rotary_emb=ref_rotary_emb,
        router_hidden_states=router_hidden_states,
        router_temb=router_temb,
        router_rotary_emb=router_rotary_emb,
        model_config=model_config,
    )
    
    # Add ControlNet signal as residual (final step)
    # Note: ControlNet signal should only guide the image representation (hidden_states)
    # and not the text representation (encoder_hidden_states)
    if controlnet_block_sample is not None:
        hidden_states = (
            hidden_states
             + controlnet_block_sample
        )

    return encoder_hidden_states, hidden_states, ref_hidden_states, router_hidden_states

def merged_single_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    ref_hidden_states: torch.FloatTensor = None,
    ref_temb: torch.FloatTensor = None,
    ref_rotary_emb=None,
    router_hidden_states: torch.FloatTensor = None,
    router_temb: torch.FloatTensor = None,
    router_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
    controlnet_single_block_sample: Optional[torch.Tensor] = None,
):
    """
    Merged single block forward function that combines AnyStory and ControlNet logic.
    
    This function:
    1. Executes the full AnyStory single block logic first
    2. Adds the ControlNet signal as a residual at the end
    """
    use_ref_cond = ref_hidden_states is not None
    use_router = router_hidden_states is not None
    
    # Execute AnyStory single block logic first
    hidden_states, ref_hidden_states, router_hidden_states = single_block_forward(
        self,
        hidden_states=hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
        ref_hidden_states=ref_hidden_states,
        ref_temb=ref_temb,
        ref_rotary_emb=ref_rotary_emb,
        router_hidden_states=router_hidden_states,
        router_temb=router_temb,
        router_rotary_emb=router_rotary_emb,
        model_config=model_config,
    )
    
    # Add ControlNet signal as residual (final step)
    # Note: ControlNet signal should only guide the image representation (hidden_states)
    if controlnet_single_block_sample is not None:
        hidden_states = hidden_states + controlnet_single_block_sample
    
    return hidden_states, ref_hidden_states, router_hidden_states

# Step 2: Create a Merged transformer_forward Function
def merged_transformer_forward(
    self,  # 'self' here refers to the transformer instance
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    # AnyStory specific
    redux_hidden_states: torch.Tensor = None,
    redux_ids: torch.Tensor = None,
    ref_hidden_states: torch.Tensor = None,
    ref_ids: torch.Tensor = None,
    router_hidden_states: torch.Tensor = None,
    router_ids: torch.Tensor = None,
    model_config: Optional[Dict[str, Any]] = {},
    # Inpainting specific
    controlnet_block_samples: Optional[List[torch.Tensor]] = None,
    controlnet_single_block_samples: Optional[List[torch.Tensor]] = None,
    # General
    return_dict: bool = True,
):
    """
    Merged transformer forward function that combines AnyStory and ControlNet logic.
    
    This function:
    1. Starts with AnyStory transformer logic
    2. Modifies the block calls to use merged block functions
    3. Passes ControlNet samples to the appropriate blocks
    4. Implements the correct data flow (cat and slice operations)
    """
    
    # Handle joint attention kwargs (from original transformer_flux.py)
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )
    
    # Prepare parameters (same as AnyStory)
    # self = transformer

    # (
    #     hidden_states,
    #     encoder_hidden_states,
    #     pooled_projections,
    #     timestep,
    #     img_ids,
    #     txt_ids,
    #     guidance,
    #     joint_attention_kwargs,
    #     return_dict,
    # ) = prepare_params(**params)

    # # required parameters calculated internally
    # model_config["txt_seq_len"] = encoder_hidden_states.shape[1]
    # model_config["img_seq_len"] = hidden_states.shape[1]
    # model_config["redux_seq_len"] = redux_hidden_states.shape[1] if redux_hidden_states is not None else 0
    # model_config["ref_seq_len"] = ref_hidden_states.shape[1] if ref_hidden_states is not None else 0
    # model_config["router_seq_len"] = router_hidden_states.shape[1] if router_hidden_states is not None else 0

    use_ref_cond = ref_hidden_states is not None and not model_config.get("use_ref_cache", False)
    use_router = router_hidden_states is not None

    if redux_hidden_states is not None and redux_ids is not None:
        print(f"txt_ids.shape line 291, in merged_transformer_forward: {txt_ids.shape}")
        print(f"redux_ids.shape line 291, in merged_transformer_forward: {redux_ids.shape}")
        encoder_hidden_states = torch.cat([encoder_hidden_states, redux_hidden_states], dim=1)
        txt_ids = torch.cat([txt_ids, redux_ids], dim=0)
    
    # required parameters calculated internally AFTER concatenation
    model_config["txt_seq_len"] = encoder_hidden_states.shape[1]
    model_config["img_seq_len"] = hidden_states.shape[1]
    model_config["redux_seq_len"] = redux_hidden_states.shape[1] if redux_hidden_states is not None else 0
    model_config["ref_seq_len"] = ref_hidden_states.shape[1] if ref_hidden_states is not None else 0
    model_config["router_seq_len"] = router_hidden_states.shape[1] if router_hidden_states is not None else 0
    
    # Update router_seq_len to account for the fact that router conditions are concatenated to query/key/value
    # in the attention processor, which affects the actual sequence length
    if router_hidden_states is not None and use_router:
        # The router sequence is added to the query/key/value in attention processor
        # So the actual router_seq_len should be the same as the router_hidden_states.shape[1]
        model_config["router_seq_len"] = router_hidden_states.shape[1]
        print(f"Updated router_seq_len: {model_config['router_seq_len']}")
        print(f"router_hidden_states.shape: {router_hidden_states.shape}")
    
    # Debug prints for sequence lengths
    print(f"model_config sequence lengths:")
    print(f"  txt_seq_len: {model_config['txt_seq_len']}")
    print(f"  img_seq_len: {model_config['img_seq_len']}")
    print(f"  redux_seq_len: {model_config['redux_seq_len']}")
    print(f"  ref_seq_len: {model_config['ref_seq_len']}")
    print(f"  router_seq_len: {model_config['router_seq_len']}")
    print(f"  use_ref_cache: {model_config.get('use_ref_cache', False)}")
    
    # Embed hidden states (from original transformer_flux.py)
    hidden_states = self.x_embedder(hidden_states)
    
    # AnyStory: Embed reference hidden states with LoRA
    with enable_lora((self.x_embedder,), ("ref_lora",), model_config.get("ref_lora_scale", 1.0)):
        ref_hidden_states = self.x_embedder(ref_hidden_states) if use_ref_cond else None
    
    # Prepare time embeddings (from original transformer_flux.py)
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None
    
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    
    # AnyStory: Prepare reference time embeddings
    ref_t = model_config.get("ref_t", 0)
    ref_temb = (
        self.time_text_embed(torch.ones_like(timestep) * ref_t * 1000, pooled_projections)
        if guidance is None else
        self.time_text_embed(torch.ones_like(timestep) * ref_t * 1000, guidance, pooled_projections)
    ) if use_ref_cond else None

    router_temb = temb if use_router else None
    
    # Embed encoder hidden states (from original transformer_flux.py)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    
    # AnyStory: Embed router hidden states with LoRA
    with enable_lora((self.context_embedder,), ("router_lora",), model_config.get("router_lora_scale", 1.0)):
        router_hidden_states = self.context_embedder(router_hidden_states) if use_router else None
    
    # Prepare rotary embeddings (from original transformer_flux.py)
    if txt_ids.shape[0] != hidden_states.shape[0]:
        txt_ids = txt_ids.expand(hidden_states.size(0), -1, -1)
    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)
    
    # AnyStory: Prepare reference rotary embeddings
    if ref_ids is not None and ref_ids.ndim == 3:
        logger.warning(
            "Passing `ref_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        print(f"ref_ids.shape: {ref_ids.shape}")
        ref_ids = ref_ids[0]

    ref_rotary_emb = self.pos_embed(ref_ids) if use_ref_cond else None

    if router_ids is not None and router_ids.ndim == 3:
        logger.warning(
            "Passing `router_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )

        print(f"router_ids.shape: {router_ids.shape}")
        router_ids = router_ids[0]
    
    router_rotary_emb = self.pos_embed(router_ids) if use_router else None
    
    # Process transformer blocks with merged logic and gradient checkpointing
    for index_block, block in enumerate(self.transformer_blocks):
        # Step 5: Implement correct ControlNet sample indexing (from original transformer_flux.py)
        current_control_sample = None
        if controlnet_block_samples is not None:
            # Calculate the mapping interval
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            # Get the correct sample index
            control_sample_index = index_block // interval_control
            # Ensure the index is valid
            if control_sample_index < len(controlnet_block_samples):
                current_control_sample = controlnet_block_samples[control_sample_index]
        
        if self.training and self.gradient_checkpointing:
            # Gradient checkpointing logic (from original transformer_flux.py)
            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)
                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            
            (encoder_hidden_states, hidden_states, ref_hidden_states, router_hidden_states) = torch.utils.checkpoint.checkpoint(
                create_custom_forward(merged_block_forward),
                self=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                ref_hidden_states=ref_hidden_states,
                ref_temb=ref_temb,
                ref_rotary_emb=ref_rotary_emb,
                router_hidden_states=router_hidden_states,
                router_temb=router_temb,
                router_rotary_emb=router_rotary_emb,
                model_config=model_config,
                controlnet_block_sample=current_control_sample,
                **ckpt_kwargs,
            )
        else:
            # Regular forward call
            (encoder_hidden_states, hidden_states, ref_hidden_states, router_hidden_states) = merged_block_forward(
                block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                ref_hidden_states=ref_hidden_states,
                ref_temb=ref_temb,
                ref_rotary_emb=ref_rotary_emb,
                router_hidden_states=router_hidden_states,
                router_temb=router_temb,
                router_rotary_emb=router_rotary_emb,
                model_config=model_config,
                controlnet_block_sample=current_control_sample,
            )
    
    # Step 2A: Add concatenation after transformer blocks (from original transformer_flux.py)
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    
    # Process single transformer blocks with merged logic and gradient checkpointing
    for index_single_block, single_block in enumerate(self.single_transformer_blocks):
        # Step 5: Implement correct ControlNet sample indexing (from original transformer_flux.py)
        current_control_single_sample = None
        if controlnet_single_block_samples is not None:
            # Calculate the mapping interval
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            # Get the correct sample index
            control_sample_index = index_single_block // interval_control
            # Ensure the index is valid
            if control_sample_index < len(controlnet_single_block_samples):
                current_control_single_sample = controlnet_single_block_samples[control_sample_index]
        
        if self.training and self.gradient_checkpointing:
            # Gradient checkpointing logic (from original transformer_flux.py)
            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)
                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            
            (hidden_states, ref_hidden_states, router_hidden_states) = torch.utils.checkpoint.checkpoint(
                create_custom_forward(merged_single_block_forward),
                self=single_block,
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                ref_hidden_states=ref_hidden_states,
                ref_temb=ref_temb,
                ref_rotary_emb=ref_rotary_emb,
                router_hidden_states=router_hidden_states,
                router_temb=router_temb,
                router_rotary_emb=router_rotary_emb,
                model_config=model_config,
                controlnet_single_block_sample=current_control_single_sample,
                **ckpt_kwargs,
            )
        else:
            # Regular forward call
            (hidden_states, ref_hidden_states, router_hidden_states) = merged_single_block_forward(
                single_block,
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                ref_hidden_states=ref_hidden_states,
                ref_temb=ref_temb,
                ref_rotary_emb=ref_rotary_emb,
                router_hidden_states=router_hidden_states,
                router_temb=router_temb,
                router_rotary_emb=router_rotary_emb,
                model_config=model_config,
                controlnet_single_block_sample=current_control_single_sample,
            )
    
    # Step 2B: Add slicing after single transformer blocks (from original transformer_flux.py)
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
    
    # Step 3: Restore the correct final output layers (from original transformer_flux.py)
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    
    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)
    
    if not return_dict:
        return (output,)
    
    from diffusers.models.transformer_2d import Transformer2DModelOutput
    return Transformer2DModelOutput(sample=output)

class UnifiedStoryInpaintPipeline(DiffusionPipeline, FluxLoraLoaderMixin):
    """
    Unified pipeline that combines AnyStory character consistency with Flux ControlNet Inpainting.
    
    This pipeline can:
    1. Generate images with consistent characters using reference images (AnyStory)
    2. Inpaint specific regions using masks (ControlNet Inpainting)
    3. Combine both capabilities for complex image generation tasks
    """
    
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["siglip_image_encoder", "siglip_image_processor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        controlnet: FluxControlNetModel,
        redux_embedder: AnyStoryReduxImageEncoder,
        router_embedder: AnyStoryReduxImageEncoder,
        siglip_image_encoder=None,
        siglip_image_processor=None,
        ref_size=512,
        torch_dtype=torch.bfloat16,
    ):
        super().__init__()
        
        # Register all components
        self.register_modules(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            controlnet=controlnet,
            redux_embedder=redux_embedder,
            router_embedder=router_embedder,
            siglip_image_encoder=siglip_image_encoder,
            siglip_image_processor=siglip_image_processor,
        )
        
        # Set up AnyStory attention processor
        self._setup_anystory_attention_processor()
        
        # Default parameters
        self.default_sample_size = 64
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels))
            if hasattr(self, "vae") and self.vae is not None
            else 16
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_resize=True, do_convert_rgb=True, do_normalize=True)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_resize=True,
            do_convert_grayscale=True,
            do_normalize=False,
            do_binarize=True,
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length
            if hasattr(self, "tokenizer") and self.tokenizer is not None
            else 77
        )
        self.ref_size = ref_size
        self.torch_dtype = torch_dtype

    def _setup_anystory_attention_processor(self):
        """Set up custom AnyStory attention processor for the transformer."""
        
        def create_custom_processor(model):
            return AnyStoryFluxAttnProcessor2_0(
                hidden_size=(
                    model.config.attention_head_dim *
                    model.config.num_attention_heads
                ),
                router_lora_rank=128,
                router_lora_bias=True,
            ).to(device=model.device, dtype=model.dtype)
        
        attn_procs_transformer = {}
        for name in self.transformer.attn_processors.keys():
            if name.endswith("attn.processor"):
                attn_procs_transformer[name] = create_custom_processor(self.transformer)
        self.transformer.set_attn_processor(attn_procs_transformer)

    @property
    def do_classifier_free_guidance(self):
        """Check if classifier-free guidance is enabled."""
        return self._guidance_scale > 1.0

    @property
    def guidance_scale(self):
        """Get the guidance scale."""
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        """Get joint attention kwargs."""
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        """Get number of timesteps."""
        return self._num_timesteps

    @property
    def interrupt(self):
        """Check if generation is interrupted."""
        return self._interrupt

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer_2.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(
            text_input_ids.to(device), output_hidden_states=False
        )[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        )

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier-free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                negative prompt to be encoded
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                negative prompt to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `negative_prompt` is
                used in all text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if do_classifier_free_guidance:
            # 处理 negative prompt
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            
            negative_pooled_prompt_embeds = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
        else:
            negative_pooled_prompt_embeds = None
            negative_prompt_embeds = None            

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(
            device=device, dtype=self.text_encoder.dtype
        )

        return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds, text_ids 

    def prepare_image_with_mask(
        self,
        image,
        mask,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance = False,
    ):
        # Prepare image
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt
        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)

        # Prepare mask
        if isinstance(mask, torch.Tensor):
            pass
        else:
            mask = self.mask_processor.preprocess(mask, height=height, width=width)
        mask = mask.repeat_interleave(repeat_by, dim=0)
        mask = mask.to(device=device, dtype=dtype)

        # Get masked image
        masked_image = image.clone()
        masked_image[(mask > 0.5).repeat(1, 3, 1, 1)] = -1

        # Encode to latents
        image_latents = self.vae.encode(masked_image.to(self.vae.dtype)).latent_dist.sample()
        image_latents = (
            image_latents - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor
        image_latents = image_latents.to(dtype)

        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor * 2, width // self.vae_scale_factor * 2)
        )
        mask = 1 - mask

        control_image = torch.cat([image_latents, mask], dim=1)

        # Pack cond latents
        packed_control_image = self._pack_latents(
            control_image,
            batch_size * num_images_per_prompt,
            control_image.shape[1],
            control_image.shape[2],
            control_image.shape[3],
        )
        
        if do_classifier_free_guidance:
            packed_control_image = torch.cat([packed_control_image] * 2)

        return packed_control_image, height, width

    def encode_redux_maybe_with_router_condition(self, image, mask, enable_router=False):
        """Encode reference image for AnyStory redux conditioning."""
        if self.siglip_image_processor is None or self.siglip_image_encoder is None:
            raise ValueError("SigLIP components are required for AnyStory conditioning")
        
        # Preprocess image
        image = self.siglip_image_processor.preprocess(images=image.convert("RGB"), return_tensors="pt").pixel_values
        mask = self.siglip_image_processor.preprocess(images=mask.convert("RGB"), resample=0, do_normalize=False,
                                                      return_tensors="pt").pixel_values
        image = (image * mask).to(device=self.device, dtype=self.torch_dtype)
        mask = mask.mean(dim=1, keepdim=True).to(device=self.device, dtype=self.torch_dtype)

        # encoding
        siglip_output = self.siglip_image_encoder(image).last_hidden_state
        s = self.siglip_image_encoder.vision_model.embeddings.image_size \
            // self.siglip_image_encoder.vision_model.embeddings.patch_size
        mask = (F.adaptive_avg_pool2d(mask, output_size=(s, s)) > 0).to(dtype=self.torch_dtype)
        mask = mask.flatten(2).transpose(1, 2)  # bs, 729, 1
        redux_embeds = self.redux_embedder(siglip_output, mask)  # [bs, 81, 4096]
        redux_ids = torch.zeros((redux_embeds.shape[1], 3), device=self.device, dtype=self.torch_dtype)

        if enable_router:
            router_embeds = self.router_embedder(siglip_output, mask)  # [bs, 1, 4096]
            router_ids = torch.zeros((router_embeds.shape[1], 3), device=self.device, dtype=self.torch_dtype)
        else:
            router_embeds = router_ids = None

        return (redux_embeds, redux_ids), (router_embeds, router_ids)

    def encode_ref_condition(self, image, mask, position_delta=None):
        """Encode reference image for AnyStory reference conditioning."""
        # Preprocess image
        image = self.image_processor.preprocess(
            image.convert("RGB"), height=self.ref_size, width=self.ref_size
        )
        mask = mask.convert("L").resize((self.ref_size, self.ref_size), resample=0)
        mask = torch.from_numpy(np.array(mask)).permute(0, 1).float() / 255.0
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]

        image = (image * mask).to(device=self.device, dtype=self.transformer.dtype)
        mask = mask.to(device=self.device, dtype=self.transformer.dtype)

        # Encode with VAE
        latent = self.vae.encode(image).latent_dist.sample()
        latent = (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        ref_embeds = self._pack_latents(latent, *latent.shape)
        ref_ids = self._prepare_latent_image_ids(
            latent.shape[0],
            latent.shape[2] // 2,
            latent.shape[3] // 2,
            self.device,
            self.transformer.dtype,
        )

        if position_delta is None:
            position_delta = [0, -self.ref_size // 16]  # width shift
        ref_ids[:, 1] += position_delta[0]
        ref_ids[:, 2] += position_delta[1]

        s = self.ref_size // 16
        ref_masks = (F.adaptive_avg_pool2d(mask, output_size=(s, s)) > 0).to(self.device, dtype=self.transformer.dtype)
        ref_masks = ref_masks.flatten(2).transpose(1, 2)  # bs, 1024, 1

        return (ref_embeds, ref_ids), ref_masks

    # Copied from diffusers.pipelines.flux.pipeline_flux.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype
            )
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, height, width
        )

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )

        return latents, latent_image_ids

    def _prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )

        return latents

    def _unpack_latents(self, latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(
            batch_size, channels // (2 * 2), height * 2, width * 2
        )

        return latents

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        """Check and validate inputs."""
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_on_step_end_tensor_inputs is not None and
            not all(k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs)):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and pooled_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `pooled_prompt_embeds`: {pooled_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if max_sequence_length is not None and max_sequence_length > self.tokenizer_2.model_max_length:
            raise ValueError(
                f"The length of the `prompt` is {max_sequence_length}, and the `max_sequence_length`"
                f" {max_sequence_length} thus cannot be set higher than the max length"
                f" {self.tokenizer_2.model_max_length} of the tokenizer. "
                f"Please reduce the value of `max_sequence_length`, or reduce the length of the `prompt`."
            ) 

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        true_guidance_scale: float = 3.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        # AnyStory parameters
        ref_images: Optional[List[Image.Image]] = None,
        ref_masks: Optional[List[Image.Image]] = None,
        redux_scale=0.6,
        redux_start_at=0.1,
        redux_end_at=0.3,
        ref_shift: float = 0.0,        
        enable_ref_mask: bool = True,
        ref_start_at: float = 0.0,
        ref_end_at: float = 1.0,
        enable_ref_cache: bool = True,
        # ControlNet parameters
        control_image: PipelineImageInput = None,
        control_mask: PipelineImageInput = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        # General parameters
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        """
        Main generation method that combines AnyStory and ControlNet Inpainting.
        
        Args:
            prompt: Text prompt for generation
            ref_images: Reference images for character consistency (AnyStory)
            ref_masks: Masks for reference images
            control_image: Control image for inpainting
            control_mask: Mask for inpainting region
            ... (other parameters)
        """
        
        # Set default dimensions
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        # Set guidance parameters
        self._guidance_scale = true_guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # Determine batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.transformer.dtype

        # Encode prompts
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        
        (
            prompt_embeds, 
            pooled_prompt_embeds, 
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            text_ids
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 在 encode_prompt 之后
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim = 0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim = 0)
            text_ids = torch.cat([text_ids, text_ids], dim = 0)
        
        # Prepare ControlNet conditions
        num_channels_latents = self.transformer.config.in_channels // 4
        
        # ControlNet
        control_image_processed = None
        if control_image is not None and control_mask is not None:
            control_image_processed, height, width = self.prepare_image_with_mask(
                image=control_image,
                mask=control_mask,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )

        # Prepare latents
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # ControlNet
        if self.do_classifier_free_guidance:
            latent_image_ids = torch.cat([latent_image_ids] * 2)

        # =================== Prepare AnyStory conditions ===================
        ref_conditions = []
        redux_conditions = []
        router_conditions = []
        ref_mask = None
        if ref_images is not None:
            # Determine router usage based on number of reference images (from storyboard.py logic)
            use_router = len(ref_images) > 1
            
            # Adjust ref_start_at based on router condition (from storyboard.py logic)
            # Only apply this logic if user hasn't specified a custom ref_start_at
            if ref_start_at == 0.0:  # Default value, apply dynamic logic
                if use_router:
                    ref_start_at = 0.09
                else:
                    ref_start_at = 0.0
            
            for i, (ref_image, ref_mask_image) in enumerate(zip(ref_images, ref_masks)):
                # Encode redux condition with dynamic router usage
                redux_with_router_condition = self.encode_redux_maybe_with_router_condition(ref_image, ref_mask_image, enable_router=use_router)
                redux_conditions.append(redux_with_router_condition[0])
                if use_router:
                    router_conditions.append(redux_with_router_condition[1])
                
                # Encode reference condition
                ref_condition, ref_mask_tensor = self.encode_ref_condition(ref_image, ref_mask_image)
                ref_conditions.append(ref_condition)
                if enable_ref_mask:
                    if ref_mask is None:
                        ref_mask = ref_mask_tensor
                    else:
                        ref_mask = torch.cat([ref_mask, ref_mask_tensor], dim=1)

        redux_embeds, redux_ids = ([] for _ in range(2))
        use_redux_cond = redux_conditions is not None and len(redux_conditions) > 0
        if use_redux_cond:
            for redux_condition in redux_conditions:
                tokens, ids = redux_condition
                redux_embeds.append(tokens * redux_scale)  # [bs, 81, 4096]
                redux_ids.append(ids)  # [81, 3]
            redux_embeds = torch.cat(redux_embeds, dim=1)
            redux_ids = torch.cat(redux_ids, dim=0)

        ref_latents, ref_ids = ([] for _ in range(2))
        use_ref_cond = ref_conditions is not None and len(ref_conditions) > 0
        if use_ref_cond:
            for ref_condition in ref_conditions:
                tokens, ids = ref_condition
                ref_latents.append(tokens)  # [bs, 1024, 4096]
                ref_ids.append(ids)  # [1024, 3]
            ref_latents = torch.cat(ref_latents, dim=1)
            ref_ids = torch.cat(ref_ids, dim=0)

        router_embeds, router_ids = ([] for _ in range(2))
        use_router = router_conditions is not None and len(router_conditions) > 0
        if use_router:
            for router_condition in router_conditions:
                tokens, ids = router_condition
                router_embeds.append(tokens)  # [bs, 81, 4096]
                router_ids.append(ids)  # [81, 3]
            router_embeds = torch.cat(router_embeds, dim=1)
            router_ids = torch.cat(router_ids, dim=0)

        num_conds = 0
        if use_redux_cond or use_ref_cond:
            num_conds = len(redux_conditions) or len(ref_conditions)

        # =================================================================

        # Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # Main denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                # Broadcast timestep
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

                # Handle guidance
                if self.transformer.config.guidance_embeds:
                    guidance = torch.tensor([guidance_scale], device=device)
                    guidance = guidance.expand(latent_model_input.shape[0])
                else:
                    guidance = None

                # handle conditions AnyStory for this timestep
                redux_start_step = int(num_inference_steps * redux_start_at + 0.5)
                redux_end_step = int(num_inference_steps * redux_end_at + 0.5)
                ref_start_step = int(num_inference_steps * ref_start_at + 0.5)
                ref_end_step = int(num_inference_steps * ref_end_at + 0.5)

                act_redux_cond = use_redux_cond and redux_start_step <= i < redux_end_step
                act_ref_cond = use_ref_cond and ref_start_step <= i < ref_end_step

                # Model configuration for AnyStory
                model_config = {}
                model_config["cache_ref"] = act_ref_cond and enable_ref_cache and i == ref_start_step
                model_config["use_ref_cache"] = act_ref_cond and enable_ref_cache and (
                    ref_start_step < i < ref_end_step
                )
                model_config["ref_shift"] = ref_shift
                model_config["ref_mask"] = ref_mask
                model_config["num_conds"] = num_conds if act_redux_cond or act_ref_cond else 0

                # Prepare AnyStory reference conditions
                redux_hidden_states = redux_embeds if act_redux_cond else None
                redux_ids = redux_ids if act_redux_cond else None
                ref_hidden_states = ref_latents if act_ref_cond else None
                ref_ids = ref_ids if act_ref_cond else None
                router_hidden_states = router_embeds if use_router and (act_redux_cond or act_ref_cond) else None
                router_ids = router_ids if use_router and (act_redux_cond or act_ref_cond) else None

                # Prepare ControlNet conditions
                if control_image_processed is not None:
                    (
                        controlnet_block_samples, 
                        controlnet_single_block_samples
                    ) = self.controlnet(
                        hidden_states=latent_model_input,
                        controlnet_cond=control_image_processed,
                        pos_embed_module=self.transformer.pos_embed,
                        conditioning_scale=controlnet_conditioning_scale,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )
                else:
                    controlnet_block_samples = None
                    controlnet_single_block_samples = None

                # Call merged transformer with all conditions
                noise_pred = merged_transformer_forward(
                    self.transformer,
                    # AnyStory specific
                    redux_hidden_states=redux_hidden_states,
                    redux_ids=redux_ids,
                    ref_hidden_states=ref_hidden_states,
                    ref_ids=ref_ids,
                    router_hidden_states=router_hidden_states,
                    router_ids=router_ids,
                    model_config=model_config,
                    # Inpainting specific
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    # General
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # Apply classifier-free guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + true_guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Update latents
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                # Handle callbacks
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # Update progress
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # Decode final latents
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            latents = latents.to(self.vae.dtype)

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Clean up
        self.maybe_free_model_hooks()

        # Clear AnyStory cache
        for attn_processor in self.transformer.attn_processors.values():
            if isinstance(attn_processor, AnyStoryFluxAttnProcessor2_0):
                for values in attn_processor.ref_bank.values():
                    del values
                attn_processor.ref_bank = {}
        torch.cuda.empty_cache()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image) 