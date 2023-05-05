from __future__ import annotations

import collections
import os
import re

import torch
from diffusers.utils import is_xformers_available
from transformers import PretrainedConfig

from dreambooth import shared  # noqa
from dreambooth.dataclasses.db_config import DreamboothConfig  # noqa
from dreambooth.utils.utils import cleanup  # noqa
from rotary_embedding_torch import RotaryEmbedding
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import Attention

if is_xformers_available():
    import xformers
    import xformers.ops
    from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
else:
    MemoryEfficientAttentionFlashAttentionOp = None
    xformers = None


db_config = None
checkpoints_list = {}
checkpoint_alisases = {}
checkpoints_loaded = collections.OrderedDict()

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(shared.models_path, model_dir))

LORA_SHARED_SRC_CREATE = " <create new>"

def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


class CheckpointInfo:
    def __init__(self, filename):
        self.filename = filename
        abspath = os.path.abspath(filename)

        if shared.cmd_opts.ckpt_dir is not None and abspath.startswith(shared.cmd_opts.ckpt_dir):
            name = abspath.replace(shared.cmd_opts.ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(filename)

        if name.startswith(os.sep) or name.startswith(os.sep):
            name = name[1:]

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace(os.sep, "_"))[0]
        self.hash = model_hash(filename)

        self.sha256 = hashes.sha256_from_cache(self.filename, "checkpoint/" + name)
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = name if self.shorthash is None else f'{name} [{self.shorthash}]'

        self.ids = [self.hash, self.model_name, self.title, name, f'{name} [{self.hash}]'] + (
            [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]'] if self.shorthash else [])

    def register(self):
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_alisases[id] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, "checkpoint/" + self.name)
        if self.sha256 is None:
            return

        self.shorthash = self.sha256[0:10]

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]']

        checkpoints_list.pop(self.title)
        self.title = f'{self.name} [{self.shorthash}]'
        self.register()

        return self.shorthash

class RotaryAttentionProcessor:
    def __init__(self, rotary_emb=None, use_xformers=False):
        self.rotary_emb = rotary_emb
        self.attention_op = MemoryEfficientAttentionFlashAttentionOp
        self.use_xformers = use_xformers

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)

        query =  self.rotary_emb.rotate_queries_or_keys(query)
        key = self.rotary_emb.rotate_queries_or_keys(key)
    
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        
        value = attn.head_to_batch_dim(value)

        if self.use_xformers and self.attention_op is not None:
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
            )
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
        
def list_models():
    checkpoints_list.clear()
    checkpoint_alisases.clear()
    model_list = modelloader.load_models(model_path=model_path, command_path=shared.cmd_opts.ckpt_dir,
                                         ext_filter=[".ckpt", ".safetensors"], ext_blacklist=[".vae.safetensors"])

    cmd_ckpt = shared.cmd_opts.ckpt
    if os.path.exists(cmd_ckpt):
        checkpoint_info = CheckpointInfo(cmd_ckpt)
        checkpoint_info.register()

        shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}",
              file=sys.stderr)

    for filename in model_list:
        checkpoint_info = CheckpointInfo(filename)
        checkpoint_info.register()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_db_models():
    output = [""]
    out_dir = shared.dreambooth_models_path
    if os.path.exists(out_dir):
        for item in os.listdir(out_dir):
            if os.path.isdir(os.path.join(out_dir, item)):
                output.append(item)
    return output


def get_shared_models():
    output = ["", LORA_SHARED_SRC_CREATE]
    out_dir = os.path.join(shared.models_path, "diffusers")
    if os.path.exists(out_dir):
        for item in os.listdir(out_dir):
            if os.path.isdir(os.path.join(out_dir, item)):
                output.append(item)
    return output


def get_lora_models(config: DreamboothConfig = None):
    output = [""]
    if config is None:
        config = shared.db_model_config
    if config is not None:
        lora_dir = os.path.join(config.model_dir, "loras")
        if os.path.exists(lora_dir):
            files = os.listdir(lora_dir)
            for file in files:
                if os.path.isfile(os.path.join(lora_dir, file)):
                    if ".pt" in file and "_txt.pt" not in file:
                        output.append(file)
    return output


def get_sorted_lora_models(config: DreamboothConfig = None):
    models = get_lora_models(config)

    def get_iteration(name: str):
        regex = re.compile(r'.*_(\d+)\.pt$')
        match = regex.search(name)
        return int(match.group(1)) if match else 0

    return sorted(models, key=lambda x: get_iteration(x))


def get_model_snapshots(config: DreamboothConfig = None):
    snaps = [""]
    if config is None:
        config = shared.db_model_config
    if config is not None:
        snaps_dir = os.path.join(config.model_dir, "checkpoints")
        if os.path.exists(snaps_dir):
            for file in os.listdir(snaps_dir):
                if os.path.isdir(os.path.join(snaps_dir, file)):
                    rev_parts = file.split("-")
                    if rev_parts[0] == "checkpoint" and len(rev_parts) == 2:
                        snaps.append(rev_parts[1])
    return snaps


def unload_system_models():
    try:
        import modules.shared
        if modules.shared.sd_model is not None:
            modules.shared.sd_model.to("cpu")
        for former in modules.shared.face_restorers:
            try:
                former.send_model_to("cpu")
            except:
                pass
        cleanup()
    except:
        pass


def reload_system_models():
    try:
        import modules.shared
        if modules.shared.sd_model is not None:
            modules.shared.sd_model.to(shared.device)
        print("Restored system models.")
    except:
        pass


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


# from modules.sd_models import CheckpointInfo

def get_checkpoint_match(search_string):
    try:
        from modules import sd_models
        for info in sd_models.checkpoints_list.values():
            if search_string in info.title or search_string in info.model_name or search_string in info.filename:
                return info
    except:
        pass
    return None


def disable_safe_unpickle():
    try:
        from modules import shared as auto_shared
        auto_shared.cmd_opts.disable_safe_unpickle = True
    except:
        pass


def enable_safe_unpickle():
    try:
        from modules import shared as auto_shared
        auto_shared.cmd_opts.disable_safe_unpickle = False
    except:
        pass


def xformerify(obj):
    if shared.db_model_config.attention is not "xformers":
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            print("Enabling SDP")
            obj.set_attn_processor(AttnProcessor2_0())
            return
        except:
            pass
    else:
        try:
            print("Enabling xformers memory efficient attention for unet")
            obj.enable_xformers_memory_efficient_attention()
        except ModuleNotFoundError:
            print("xformers not found, using default attention")


def torch2ify(unet):
    if hasattr(torch, 'compile'):
        try:
            unet = torch.compile(unet, mode="max-autotune", fullgraph=False)
            print("Compiled unet")
        except:
            pass
    return unet

def handle_rotary_attention(model, use_rotary_embed=False, use_xformers=False):
    applied = 0
    if use_rotary_embed:
        rotary_emb = RotaryEmbedding(dim=8).cuda()
        model.set_attn_processor(
            RotaryAttentionProcessor(rotary_emb=rotary_emb)
        )
        for m in model.modules():
            if isinstance(m, torch.nn.ModuleList):
                for module in m:
                    if isinstance(module, BasicTransformerBlock):
                        applied += 1              
        if applied > 0: print(f"{applied} Attention layers have rotary attention.")