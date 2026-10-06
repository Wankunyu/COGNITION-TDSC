    
                    
                                     
                                                                 

    
import os, re, io, json, time, base64, argparse, random, pathlib, mimetypes, platform
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import yaml

from .revision_artifacts import (
    AttemptRecord,
    PromptConfig,
    RevisionArtifactWriter,
    RunManifest,
    collect_code_revision,
    collect_dependency_versions,
    sha256_file,
    sha256_text,
    utc_now,
)

import tqdm
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs): return x


import openai
import anthropic

            
try:
    from .experiments_helper import SimpleErrorCollector, load_tasks_from_ground_truth
    ERROR_COLLECTOR_AVAILABLE = True
except ImportError:
    ERROR_COLLECTOR_AVAILABLE = False
    print("⚠️ experiments_helper not found，error analysis unavailable")


    
            
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
IMAGE_EXTENSIONS = {ext.lower() for ext in IMG_EXTS}
IMAGE_EXTENSIONS = {ext.lower() for ext in IMG_EXTS}

                                                     
REASONING_INSTRUCTION = """
For research purposes, please provide a complete trace of your cognitive process when solving this problem. Your response should include:

- **Raw Observation**: Describe everything you perceive/understand from the input
- **Attention Focus**: What elements draw your attention first and why
- **Initial Hypotheses**: Your immediate interpretations or assumptions

- **Problem Decomposition**: Break down the task into sub-components
- **Feature Extraction**: List all relevant features, patterns, or regularities you identify
- **Uncertainty Mapping**: Explicitly mark any ambiguous or unclear aspects (with confidence levels 0-100%)

Please number each reasoning step and include:
- The specific operation or inference being made
- The evidence or logic supporting this step
- Alternative interpretations considered (even if rejected)
- Confidence level for this step
- Any assumptions being made

- **Potential Failure Points**: Where might your reasoning go wrong?
- **Consistency Checks**: How do you verify internal consistency?
- **Alternative Paths**: What other approaches did you consider but not pursue?

- **Candidate Answers**: List all possibilities considered
- **Selection Rationale**: Why you chose the final answer
- **Overall Confidence**: Your certainty level in the final answer

**Important Instructions**:
- Do not skip steps that seem "obvious" - document everything
- Include failed attempt and backtracking
- Show intermediate results even if incorrect
- Think aloud - as if debugging your own thinking
- If you're guessing, explicitly state so and explain why

This detailed trace is for academic research on AI reasoning transparency. Please be as verbose and explicit as possible about your internal process.
"""

def _list_images_in_dir(d: str) -> list[str]:
    if not os.path.isdir(d): 
        return []
    return sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )

def _maybe_parse_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _stem_glob(type_dir: str, pid: str) -> str | None:
    """Try to find image file with same stem (filename without extension) in directory."""
    stem = os.path.splitext(pid)[0]
    for ext in IMG_EXTS:
        cand = os.path.join(type_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    return None


def load_secrets(path: str) -> dict:
    """
    Load secrets/config file (secrets.yaml or secrets.json).
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Secrets file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f) or {}


def load_prompts(path: Optional[str]) -> Dict[str, str]:
    """
    Load prompt override file (yaml/json). Returns {task_type or 'default': prompt_text}.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found prompt file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) if path.endswith(".json") else (yaml.safe_load(f) or {})
             
    return {str(k): str(v) for k, v in (data or {}).items() if isinstance(v, str)}


def estimate_cost(provider: str, model: str, tokens_in: Optional[int], tokens_out: Optional[int], secrets: dict) -> Optional[float]:
    """
    Estimate per-question cost based on pricing section in secrets.yaml (USD per 1k tokens).
    """
    try:
        pricing = (secrets.get("pricing") or {}).get(provider, {})
        p = pricing.get(model) or pricing.get(model.lower())
        if not p:
            print(f"[WARNING] No pricing found for provider='{provider}', model='{model}'")
            return None
        def _cost(toks, per_1k):
            if toks is None or per_1k is None: return 0.0
            return (toks / 1000.0) * float(per_1k)
        return round(_cost(tokens_in, p.get("in_per_1k")) + _cost(tokens_out, p.get("out_per_1k")), 6)
    except Exception as e:
        print(f"[ERROR] estimate_cost failed: provider={provider}, model={model}, error={e}")
        import traceback
        traceback.print_exc()
        return None


def extract_json(text: str) -> Optional[dict]:
    """
    Lightweight JSON extraction：cleancodeblocks、attemptdirectly loads，failedthenfallbackto“outerbrace”extraction。
    """
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I|re.M)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, (dict, list)) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, (dict, list)) else None
        except Exception:
            return None
    return None


def guess_mime(path: str) -> str:
    """
    Detect MIME type based on file content (magic bytes) instead of extension.
    This avoids API errors from mismatched file extensions.
    """
    try:
        with open(path, "rb") as f:
            header = f.read(12)                    

                                      
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"

                        
        if header[:3] == b'\xff\xd8\xff':
            return "image/jpeg"

                               
        if header[:6] in (b'GIF87a', b'GIF89a'):
            return "image/gif"

                            
        if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            return "image/webp"

                 
        if header[:2] == b'BM':
            return "image/bmp"

                                                     
        if header[:2] in (b'II\x2a\x00', b'MM\x00\x2a'):
            return "image/tiff"

    except Exception:
        pass

                                           
    m, _ = mimetypes.guess_type(path)
    if m:
        return m
    ext = os.path.splitext(path.lower())[1]
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }.get(ext, "application/octet-stream")

def _is_rect_hit(x: float, y: float, box: dict) -> bool:
    """Check if point hits rectangle (inclusive); box: {x,y,width,height} or [[x1,y1],[x2,y2]]"""
    if not box:
        return False
    if all(k in box for k in ("x","y","width","height")):
        return (box["x"] <= x <= box["x"] + box["width"] and
                box["y"] <= y <= box["y"] + box["height"])
    if isinstance(box, (list, tuple)) and len(box) == 2:
        (x1,y1),(x2,y2) = box
        return (min(x1,x2) <= x <= max(x1,x2) and
                min(y1,y2) <= y <= max(y1,y2))
    return False

def _point_dist(p: dict, q: dict) -> float:
    """Euclidean distance between two points; p/q: {"x":...,"y":...}"""
    return ((float(p["x"]) - float(q["x"]))**2 + (float(p["y"]) - float(q["y"]))**2) ** 0.5

def _normalize_points(points) -> list[dict]:
    """Normalize points to [{'x': float, 'y': float}] list; tolerate list/dict inputs."""
    if points is None:
        return []
    if isinstance(points, dict):
        points = [points]
    out = []
    for p in (points or []):
        try:
            if isinstance(p, dict):
                x, y = p.get("x"), p.get("y")
            elif isinstance(p, (list, tuple)) and len(p) == 2:
                x, y = p
            else:
                continue
            out.append({"x": float(x), "y": float(y)})
        except Exception:
            continue
    return out

def _match_points(pred_pts: list[dict], gt_pts: list[dict], tol: float) -> tuple[int, int, int]:
    """Greedy match points within tolerance; returns (matched, missing_gt, extra_pred)."""
    used_pred = set()
    matched = 0
    for g in gt_pts:
        best_idx = None
        best_dist = None
        for i, p in enumerate(pred_pts):
            if i in used_pred:
                continue
            dist = _point_dist(p, g)
            if dist <= tol and (best_dist is None or dist < best_dist):
                best_idx = i
                best_dist = dist
        if best_idx is not None:
            used_pred.add(best_idx)
            matched += 1
    missing_gt = max(0, len(gt_pts) - matched)
    extra_pred = max(0, len(pred_pts) - len(used_pred))
    return matched, missing_gt, extra_pred

def _clean_indices(xs) -> list:
    """Deduplicate/convert to int/sort index set"""
    try:
        return sorted(set(int(i) for i in (xs or [])))
    except Exception:
        return []

def _correct_as_set(correct) -> frozenset:
    """Normalize correct (i,j) to unordered set for comparison"""
    if isinstance(correct, (list, tuple)) and len(correct) == 2:
        return frozenset([int(correct[0]), int(correct[1])])
    return frozenset()

def _get_first(d: dict, *keys, default=None):
    """Get first existing key from dict by priority"""
    for k in keys:
        if k in d:
            return d[k]
    return default


def _extract_number(value) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))
    except Exception:
        return None


def _format_delta(pred: Optional[float], gt: Optional[float]) -> str:
    if pred is None or gt is None:
        return ""
    delta = pred - gt
    sign = "+" if delta > 0 else "" if delta < 0 else ""
    return f" (Δ={sign}{delta:.0f})" if delta != 0 else ""


def _describe_failure(task, parsed, raw: str) -> str:
    task_type = task.type
    gt = task.gt or {}

    if task_type == "Dice_Count":
        gt_sum = _extract_number(gt.get("sum"))
        if not isinstance(parsed, dict):
            return f"Did not return structured JSON; GT count = {gt_sum}."
        pred_value = _extract_number(parsed.get("value"))
        if pred_value is None:
            return f"Missing valid value field; GT count = {gt_sum}."
        if parsed.get("answer_type") != "number":
            return f"answer_type should be 'number', got {parsed.get('answer_type')!r}; predicted {pred_value}, GT = {gt_sum}{_format_delta(pred_value, gt_sum)}."
        if gt_sum is None:
            return f"Predicted value {pred_value}, unable to read GT count."
        if pred_value == gt_sum:
            return "Predicted count matches GT but failed validation, possibly field or type mismatch."
        if pred_value > gt_sum:
            return f"Predicted count {pred_value} exceeds GT {gt_sum}{_format_delta(pred_value, gt_sum)}, model likely double-counted or included non-top faces."
        return f"Predicted count {pred_value} below GT {gt_sum}{_format_delta(pred_value, gt_sum)}, model likely missed some top faces."

    if task_type == "Click_Order":
        if not isinstance(parsed, dict):
            return "Did not return click sequence."
        pred_pts = parsed.get("points") or []
        if not isinstance(pred_pts, list) or not pred_pts:
            return "points list empty or malformed."
        gt_pts = gt.get("points_gt") or []
        tol = float(gt.get("tolerance", 40.0))
        messages = []
        min_len = min(len(pred_pts), len(gt_pts))
        for idx in range(min_len):
            p = pred_pts[idx] if isinstance(pred_pts[idx], dict) else None
            q = gt_pts[idx] if isinstance(gt_pts[idx], dict) else None
            if not p or not q:
                messages.append(f"Click {idx+1} Missing coordinates.")
                continue
            dist = _point_dist(p, q)
            if dist > tol:
                messages.append(
                    f"Click {idx+1} distance from targets {dist:.1f}px (> tolerance {tol}); predicted ({p.get('x'):.0f},{p.get('y'):.0f}) vs GT ({q.get('x'):.0f},{q.get('y'):.0f})."
                )
        if len(pred_pts) != len(gt_pts):
            messages.append(f"Predicted count {len(pred_pts)} vs GT {len(gt_pts)} mismatch.")
        if not messages:
            messages.append("Click order error detected but details unclear, check raw response: " + (raw[:120] if raw else "(empty)"))
        return " ".join(messages)

    if task_type == "Select_Animal_Optimized":
        if not isinstance(parsed, dict):
            return "Did not return points list."
        pred_pts = _normalize_points(parsed.get("points"))
        gt_pts = _normalize_points(gt.get("targets_positions"))
        tol = float(gt.get("tolerance", 15.0))
        if not pred_pts:
            return "points list empty or malformed."
        if not gt_pts:
            return "GT points missing."
        matched, missing_gt, extra_pred = _match_points(pred_pts, gt_pts, tol)
        parts = []
        if parsed.get("answer_type") != "multi_point":
            parts.append(f"answer_type should be 'multi_point', got {parsed.get('answer_type')!r}.")
        if len(pred_pts) != len(gt_pts):
            parts.append(f"Predicted {len(pred_pts)} points vs GT {len(gt_pts)}.")
        if missing_gt:
            parts.append(f"{missing_gt} GT point(s) unmatched within {tol}px.")
        if extra_pred:
            parts.append(f"{extra_pred} predicted point(s) unmatched within {tol}px.")
        if not parts:
            parts.append("No specific mismatch detected, check raw response formatting.")
        return " ".join(parts)

    if task_type == "Patch_Select":
        if not isinstance(parsed, dict):
            return "Did not return selection indices."
        pred = set(int(i) for i in (parsed.get("indices") or []))
        gt_indices = set(int(i) for i in (gt.get("indices_gt") or gt.get("correct_patches") or []))
        missing = sorted(gt_indices - pred)
        extra = sorted(pred - gt_indices)
        parts = []
        if missing:
            parts.append(f"Missing {len(missing)} correct indices: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if extra:
            parts.append(f"Extra selected {len(extra)} incorrect indices: {extra[:8]}{'...' if len(extra) > 8 else ''}")
        if not parts:
            parts.append("No difference detected, check if raw response format meets requirements.")
        return "；".join(parts)

    if task_type == "Pick_Area":
        if not isinstance(parsed, dict):
            return "Did not return coordinate point."
        pt = parsed.get("point") or {}
        if "x" not in pt or "y" not in pt:
            return "Predicted point Missing x/y coordinate."
        x_pred = float(pt.get("x"))
        y_pred = float(pt.get("y"))
        box = gt.get("area_box")
        if not box:
            return "GT area missing, cannot analyze."
        if _is_rect_hit(x_pred, y_pred, box):
            return "Predicted point falls within GT area but still failed validation, check answer_type or JSON structure."
                   
        if isinstance(box, dict) and all(k in box for k in ("x","y","width","height")):
            x1, y1 = float(box["x"]), float(box["y"])
            x2, y2 = x1 + float(box["width"]), y1 + float(box["height"])
        elif isinstance(box, (list, tuple)) and len(box) == 2:
            (x1, y1), (x2, y2) = box
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        else:
            return "GT area format abnormal, cannot analyze."
        return (
            f"Predicted point ({x_pred:.1f},{y_pred:.1f}) not within GT area."
            f"GT range ≈ [{min(x1,x2):.1f},{min(y1,y2):.1f}] to [{max(x1,x2):.1f},{max(y1,y2):.1f}] pixels."
        )

    if task_type in ("Place_Dot", "Geometry_Click", "Misleading_Click"):
        if not isinstance(parsed, dict):
            return "Did not return coordinate point."
        pred_pt = parsed.get("point") or {}
        gt_pt = gt.get("targets_position") or {}
        tol = float(gt.get("tolerance", 15.0))
        if not ("x" in pred_pt and "y" in pred_pt):
            return "Predicted point Missing x/y coordinate."
        if not ("x" in gt_pt and "y" in gt_pt):
            return "GT coordinate missing, cannot analyze."
        dist = _point_dist(pred_pt, gt_pt)
        return (f"Predicted coordinate ({pred_pt.get('x'):.1f},{pred_pt.get('y'):.1f}) distance to GT ({gt_pt.get('x'):.1f},{gt_pt.get('y'):.1f}) is {dist:.1f}px，"
                f"exceeds tolerance {tol}px，indicating position deviates from targets area.")

    if task_type == "Image_Matching" or task_type in ("Dart_Count","Coordinates","Connect_Icon","Object_Match"):
        if not isinstance(parsed, dict):
            return "Did not return classification result."
        idx = parsed.get("index")
        gt_idx = gt.get("correct_index")
        return f"Predicted index={idx}，GT={gt_idx}，Need to check if model's selected match item among candidates is correct."

    if task_type in ("Image_Recognition","Select_Animal","Unusual_Detection","Path_Finder"):
        if not isinstance(parsed, dict):
            return "Did not return multi-selection result."
        pred = sorted(_clean_indices(parsed.get("indices", [])))
        gt_indices = sorted(_clean_indices(gt.get("indices_gt", [])))
        missing = sorted(set(gt_indices) - set(pred))
        extra = sorted(set(pred) - set(gt_indices))
        parts = []
        if missing:
            parts.append(f"Missing {len(missing)} targetss: {missing[:8]}{'...' if len(missing)>8 else ''}")
        if extra:
            parts.append(f"Extra selected {len(extra)} : {extra[:8]}{'...' if len(extra)>8 else ''}")
        if not parts:
            parts.append("No specific difference identified, possibly non-compliant JSON structure.")
        return "；".join(parts)

          
    raw_preview = (raw[:120] + "...") if raw and len(raw) > 120 else (raw or "")
    return f"Task type {task_type} has no dedicated analysis, please check raw response: {raw_preview}"


               
                                            

    
class ImageCache:
    """
    Original image without any conversion/re-encoding:
    - Read raw file bytes directly and cache
    - Also cache base64 to avoid repeated encoding
    - LRU eviction for controlled memory usage
    """
    def __init__(self, max_items: int = 512):
        self.max_items = max_items
        self._order: List[str] = []
        self._bytes: Dict[str, bytes] = {}
        self._b64: Dict[str, str] = {}

    def _touch(self, k: str):
        if k in self._order:
            self._order.remove(k)
        self._order.append(k)
        while len(self._order) > self.max_items:
            old = self._order.pop(0)
            self._bytes.pop(old, None)
            self._b64.pop(old, None)

    def get_bytes(self, path: str) -> bytes:
        """
        Return raw file bytes (no image processing)
        """
        if path in self._bytes:
            self._touch(path)
            return self._bytes[path]
        with open(path, "rb") as f:
            data = f.read()
        self._bytes[path] = data
        self._touch(path)
        return data

    def get_b64(self, path: str) -> str:
        """
        Return base64 text of raw file bytes (without data: prefix)
        """
        if path in self._b64:
            self._touch(path)
            return self._b64[path]
        data = self.get_bytes(path)
        b64 = base64.b64encode(data).decode("utf-8")
        self._b64[path] = b64
        self._touch(path)
        return b64


IMG_CACHE = ImageCache(max_items=512)

               
                         

    
class ModelProvider:
    """
    Provider abstract base class: defines unified infer interface.
    """
    def __init__(self, model:str, api_key:str, **kwargs):
        self.model = model
        self.api_key = api_key
        self.timeout = float(kwargs.get("timeout_sec", 120.0))
        self.thinking_enabled = bool(kwargs.get("thinking_enabled", False))
        self.thinking_options = kwargs.get("thinking_options") or {}

    def infer(self, prompt:str, images:List[str], json_schema:Dict[str, Any],
              stream:bool=True) -> Tuple[str, Optional[Dict[str,Any]], Dict[str,Any]]:
        raise NotImplementedError


               
                      

    
class OpenAIProvider(ModelProvider):
    """
    OpenAI Provider supporting GPT-5 chat (Chat Completions) and GPT-5 reasoning
    models (Responses API) with strict JSON outputs.
    """

    CHAT_MODELS = {"gpt-5-chat-latest"}
    REASONING_MODELS = {"gpt-5", "gpt-5.1"}

    def __init__(self, model: str, api_key: str, **kwargs):
        super().__init__(model, api_key, **kwargs)
        if not self.api_key:
            raise RuntimeError("OpenAI: Please configure api_key in secrets.yaml.")
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)

        normalized = model.lower()
        if normalized in self.CHAT_MODELS:
            self.is_chat_family = True
            self.is_reasoning_family = False
        elif normalized in self.REASONING_MODELS:
            self.is_chat_family = False
            self.is_reasoning_family = True
        else:
            raise RuntimeError(
                f"OpenAIProvider: Unsupported model {model}. Currently only supports gpt-5-chat-latest, gpt-5, and gpt-5.1 (reasoning models)."
            )

                                                                       
        self.reasoning_effort = self.thinking_options.get("effort", "medium")
        self.text_verbosity = self.thinking_options.get("verbosity", "medium")
        self.use_strict_schema = bool(
            kwargs.get("strict_json_schema")
            or self.thinking_options.get("strict_json_schema", False)
        )

                                           

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a vision agent. Output ONLY a JSON object that strictly matches the given schema. "
            "If the schema contains optional field 'reasoning', you MUST fill it following the detailed cognitive tracing format provided."
        )

    @staticmethod
    def _chat_image_block(path: str) -> Dict[str, Any]:
        mime = guess_mime(path)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{IMG_CACHE.get_b64(path)}",
                "detail": "high",
            },
        }

    @staticmethod
    def _responses_image_block(path: str) -> Dict[str, Any]:
        mime = guess_mime(path)
        return {
            "type": "input_image",
            "image_url": f"data:{mime};base64,{IMG_CACHE.get_b64(path)}",
        }

    @staticmethod
    def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return default

    def _build_chat_messages(
        self,
        prompt: str,
        images: List[str],
        few_shot_examples: Optional[List],
        schema_text: str,
        reasoning_note: str,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []

        if few_shot_examples:
            for example_images, example_text in few_shot_examples:
                for img_path in example_images:
                    if os.path.exists(img_path):
                        content.append(self._chat_image_block(img_path))
                content.append({"type": "text", "text": example_text})

        for img_path in images:
            if os.path.exists(img_path):
                content.append(self._chat_image_block(img_path))

        final_prompt = (
            "Now solve this new problem:\n\n" + prompt
            if few_shot_examples
            else prompt
        )
        text_block = (
            final_prompt
            + "\n\nSchema:\n"
            + schema_text
            + (f"\n\n{reasoning_note}" if reasoning_note else "")
        )
        content.append({"type": "text", "text": text_block})

        return [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": content},
        ]

    def _build_responses_input(
        self,
        prompt: str,
        images: List[str],
        few_shot_examples: Optional[List],
        schema_text: str,
        reasoning_note: str,
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": self._system_prompt()}],
            }
        ]

        user_content: List[Dict[str, Any]] = []

        if few_shot_examples:
            for example_images, example_text in few_shot_examples:
                for img_path in example_images:
                    if os.path.exists(img_path):
                        user_content.append(self._responses_image_block(img_path))
                user_content.append({"type": "input_text", "text": example_text})

        for img_path in images:
            if os.path.exists(img_path):
                user_content.append(self._responses_image_block(img_path))

        final_prompt = (
            "Now solve this new problem:\n\n" + prompt
            if few_shot_examples
            else prompt
        )
        text_block = (
            final_prompt
            + "\n\nSchema:\n"
            + schema_text
            + (f"\n\n{reasoning_note}" if reasoning_note else "")
        )
        user_content.append({"type": "input_text", "text": text_block})

        payload.append({"role": "user", "content": user_content})
        return payload

                                      

    def infer(
        self,
        prompt: str,
        images: List[str],
        json_schema: Dict[str, Any],
        stream: bool = True,
        few_shot_examples: Optional[List] = None,
    ) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
        schema_hint = json.dumps(json_schema, ensure_ascii=False)
        has_reasoning = "reasoning" in json_schema.get("properties", {})
        reasoning_note = REASONING_INSTRUCTION if has_reasoning else ""

        start = time.perf_counter()
        ttft_ms: Optional[float] = None
        tokens_in = tokens_out = None

        try:
            if self.is_chat_family:
                messages = self._build_chat_messages(
                    prompt, images, few_shot_examples, schema_hint, reasoning_note
                )
                raw, tokens_in, tokens_out, ttft_ms = self._infer_chat(messages, stream)
            else:
                responses_input = self._build_responses_input(
                    prompt, images, few_shot_examples, schema_hint, reasoning_note
                )
                raw, tokens_in, tokens_out, ttft_ms = self._infer_responses(
                    responses_input, json_schema, stream
                )
        except Exception as e:
            raw = f"__ERROR__: {type(e).__name__}: {e}"
            ttft_ms = ttft_ms or (time.perf_counter() - start) * 1000.0

        e2e_ms = (time.perf_counter() - start) * 1000.0
        parsed = extract_json(raw)
        meta = dict(
            ttft_ms=ttft_ms or e2e_ms,
            e2e_ms=e2e_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=None,
        )
        return raw, parsed, meta

                                       

    def _infer_chat(
        self, messages: List[Dict[str, Any]], stream: bool
    ) -> Tuple[str, Optional[int], Optional[int], Optional[float]]:
        response_format = {"type": "json_object"}
        start = time.perf_counter()
        ttft_ms: Optional[float] = None
        tokens_in = tokens_out = None

        if stream:
            chunks: List[str] = []
            for event in self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=0,
                stream=True,
            ):
                delta = event.choices[0].delta.content if event.choices else None
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000.0
                    chunks.append(delta)
            raw = "".join(chunks)
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            if usage:
                tokens_in = getattr(usage, "prompt_tokens", None)
                tokens_out = getattr(usage, "completion_tokens", None)
            ttft_ms = (time.perf_counter() - start) * 1000.0

        return raw, tokens_in, tokens_out, ttft_ms

                                                            

    def _infer_responses(
        self,
        responses_input: List[Dict[str, Any]],
        json_schema: Dict[str, Any],
        stream: bool,
    ) -> Tuple[str, Optional[int], Optional[int], Optional[float]]:
        request_body: Dict[str, Any] = {
            "model": self.model,
            "input": responses_input,
            "text": {"verbosity": self.text_verbosity},
        }

        if self.thinking_enabled:
            effort = self.thinking_options.get("effort", self.reasoning_effort)
            request_body["reasoning"] = {"effort": effort}

        if self.use_strict_schema:
            request_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ExtractionSchema",
                    "schema": json_schema,
                    "strict": True,
                },
            }

        start = time.perf_counter()
        ttft_ms: Optional[float] = None
        tokens_in = tokens_out = None

        if stream:
            chunks: List[str] = []
            for event in self.client.responses.create(stream=True, **request_body):
                event_type = self._safe_get(event, "type")
                if event_type == "response.output_text.delta":
                    delta = self._safe_get(event, "delta")
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start) * 1000.0
                        chunks.append(delta)
                elif event_type == "response.completed":
                    resp_obj = self._safe_get(event, "response")
                    if resp_obj:
                        usage = self._safe_get(resp_obj, "usage")
                        if usage:
                            tokens_in = self._safe_get(usage, "input_tokens")
                            tokens_out = self._safe_get(usage, "output_tokens")
            raw = "".join(chunks)
        else:
            resp = self.client.responses.create(**request_body)
            output_chunks: List[str] = []
            outputs = self._safe_get(resp, "output")
            if outputs:
                for item in outputs:
                    content_list = self._safe_get(item, "content")
                    if not content_list:
                        continue
                    for content in content_list:
                        ctype = self._safe_get(content, "type")
                        if ctype == "output_text":
                            text = self._safe_get(content, "text")
                            if text:
                                output_chunks.append(text)
            raw = "".join(output_chunks)
            usage = self._safe_get(resp, "usage")
            if usage:
                tokens_in = self._safe_get(usage, "input_tokens")
                tokens_out = self._safe_get(usage, "output_tokens")
            ttft_ms = (time.perf_counter() - start) * 1000.0

        return raw, tokens_in, tokens_out, ttft_ms

               
                         

    
class AnthropicProvider(ModelProvider):
    """
    Anthropic (Claude): Uses messages.create + json_schema tools.
    imageas“rawbytes base64 + MIME”passed，nodoconversion。
    """
    def __init__(self, model:str, api_key:str, **kwargs):
        super().__init__(model, api_key, **kwargs)
        if not self.api_key:
            raise RuntimeError("Anthropic: Please configure api_key in secrets.yaml.")
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self._thinking_payload = None
        if self.thinking_enabled:
            try:
                budget = int(self.thinking_options.get("budget_tokens", 1024))
            except Exception:
                budget = 1024
            thinking_type = self.thinking_options.get("type", "structured")
            self._thinking_payload = {"type": thinking_type, "budget_tokens": budget}

    @staticmethod
    def _img_part(path:str)->dict:
        """
        Prepare image data for Anthropic API usage.
        Automatically compress images if base64 data exceeds 5MB to comply with API limits.
        """
        b64 = IMG_CACHE.get_b64(path)
        mime = guess_mime(path)

                                   
        MAX_SIZE_BYTES = 5 * 1024 * 1024        
        b64_size = len(b64) * 3 // 4                           

        if b64_size > MAX_SIZE_BYTES:
                    
            try:
                from PIL import Image
                import io

                        
                img = Image.open(path)

                                            
                for quality in [85, 75, 65, 55, 45, 35]:
                    buffer = io.BytesIO()

                                    
                    if img.mode in ('RGBA', 'LA', 'P'):
                                       
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        rgb_img.save(buffer, format='JPEG', quality=quality, optimize=True)
                    else:
                        img.save(buffer, format='JPEG', quality=quality, optimize=True)

                    compressed_bytes = buffer.getvalue()
                    compressed_b64 = base64.b64encode(compressed_bytes).decode('ascii')

                    if len(compressed_bytes) <= MAX_SIZE_BYTES:
                        print(f"  [Anthropic] Image {os.path.basename(path)} compressed: {b64_size / (1024 * 1024):.2f} MB → {len(compressed_bytes) / (1024 * 1024):.2f} MB (quality={quality})")
                        return {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":compressed_b64}}

                                    
                print(f"  [Anthropic] Warning: Image {os.path.basename(path)} cannot be compressed below 5MB even at lowest quality")
                return {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":compressed_b64}}

            except Exception as e:
                print(f"  [Anthropic] Image compression failed: {e}, using original image")

        return {"type":"image","source":{"type":"base64","media_type":mime,"data":b64}}

    def infer(self, prompt:str, images:List[str], json_schema:Dict[str, Any],
              stream:bool=True, few_shot_examples:Optional[List]=None)->Tuple[str, Optional[Dict[str,Any]], Dict[str,Any]]:
                                               
        has_reasoning = "reasoning" in json_schema.get("properties", {})
        reasoning_note = REASONING_INSTRUCTION if has_reasoning else ""

        blocks = []

                                                 
        if few_shot_examples:
            for example_images, example_text in few_shot_examples:
                                    
                for img_path in example_images:
                    if os.path.exists(img_path):
                        blocks.append(self._img_part(img_path))
                                         
                blocks.append({"type":"text","text": example_text})

                            
        for p in images:
            blocks.append(self._img_part(p))

                                                              
        if few_shot_examples:
            final_prompt = "Now solve this new problem:\n\n" + prompt
        else:
            final_prompt = prompt

        blocks.append({"type":"text","text": (
            final_prompt +
            (f"\n\n{reasoning_note}" if reasoning_note else "")
        )})

                                                                                   
                                                                   
        tools = [{
            "name": "submit_answer",
            "description": "Submit the answer in the required JSON format",
            "input_schema": json_schema
        }]

        start = time.perf_counter()
        tokens_in = tokens_out = None
        try:
            call_kwargs = dict(
                model=self.model,
                max_tokens=2048,
                temperature=0,
                messages=[{"role":"user","content":blocks}],
                tools=tools,
                tool_choice={"type": "tool", "name": "submit_answer"}
            )
            if self._thinking_payload:
                call_kwargs["thinking"] = self._thinking_payload
            resp = self.client.messages.create(**call_kwargs)

                                              
            raw = ""
            tool_input = None
            for c in (resp.content or []):
                if getattr(c, "type", "") == "tool_use":
                    tool_input = getattr(c, "input", None)
                    if tool_input:
                        raw = json.dumps(tool_input)
                        break

                                                           
            if not raw:
                for c in (resp.content or []):
                    if getattr(c, "type", "") == "text":
                        raw += c.text or ""

            usage = getattr(resp, "usage", None)
            if usage:
                tokens_in = getattr(usage, "input_tokens", None)
                tokens_out = getattr(usage, "output_tokens", None)
        except Exception as e:
            raw = f'__ERROR__: {type(e).__name__}: {e}'
        e2e = (time.perf_counter() - start) * 1000.0
        parsed = extract_json(raw)
        meta = dict(ttft_ms=e2e, e2e_ms=e2e, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=None)
        return raw, parsed, meta

               
                      

    
class GeminiProvider(ModelProvider):
    """
    Gemini Provider (always inline raw image bytes, does not use File API)
    - No image conversion/re-encoding: directly read raw bytes from disk.
    - parts structure: [ text, {"mime_type":..., "data": <raw bytes>}, ... ].
    - Explicitly set request_options.timeout to avoid hanging.
    """
    def __init__(self, model:str, api_key:str, **kwargs):
        super().__init__(model, api_key, **kwargs)
        if not self.api_key:
            raise RuntimeError("Gemini: Please configure api_key in secrets.yaml.")

        from google import genai
        from google.genai import types as genai_types

        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        self.client = genai.Client(**client_kwargs)
        self.genai_types = genai_types

    @staticmethod
    def sanitize_response_schema(schema: Any) -> Any:
        if isinstance(schema, dict):
            return {
                key: GeminiProvider.sanitize_response_schema(value)
                for key, value in schema.items()
                if key not in {"additionalProperties", "additional_properties"}
            }
        if isinstance(schema, list):
            return [GeminiProvider.sanitize_response_schema(value) for value in schema]
        return schema

    def infer(self, prompt:str, images:List[str], json_schema:Dict[str, Any],
              stream:bool=True, few_shot_examples:Optional[List]=None) -> Tuple[str, Optional[Dict[str,Any]], Dict[str,Any]]:
        """
        - Directly inline raw bytes (zero conversion);
        - Gemini's non-streaming response latency is more stable in most regions, stream not used here.
        - Supports few-shot learning
        """
                                               
        has_reasoning = "reasoning" in json_schema.get("properties", {})
        reasoning_note = REASONING_INSTRUCTION if has_reasoning else ""

        user_parts: List[Any] = []

                                                 
        if few_shot_examples:
            for example_images, example_text in few_shot_examples:
                                    
                for img_path in example_images:
                    if os.path.exists(img_path):
                        user_parts.append(
                            self.genai_types.Part.from_bytes(
                                data=IMG_CACHE.get_bytes(img_path),
                                mime_type=guess_mime(img_path)
                            )
                        )
                                         
                user_parts.append(
                    self.genai_types.Part.from_text(text=example_text)
                )

                            
        for p in images:
            user_parts.append(
                self.genai_types.Part.from_bytes(
                    data=IMG_CACHE.get_bytes(p), mime_type=guess_mime(p)
                )
            )

                                                              
        if few_shot_examples:
            final_prompt = "Now solve this new problem:\n\n" + prompt
        else:
            final_prompt = prompt

        user_parts.append(
            self.genai_types.Part.from_text(
                text=(
                    final_prompt +
                    "\n\nReturn ONLY valid JSON per schema:\n" + json.dumps(json_schema, ensure_ascii=False) +
                    (f"\n\n{reasoning_note}" if reasoning_note else "")
                )
            )
        )

        content = self.genai_types.Content(role="user", parts=user_parts)

        schema_obj: Any = json_schema
        try:
            schema_obj = self.genai_types.Schema(
                **GeminiProvider.sanitize_response_schema(json_schema)
            )
        except Exception:
            pass

        config: Dict[str, Any] = {
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_schema": schema_obj,
        }
        if self.thinking_enabled:
            try:
                budget = self.thinking_options.get("thinking_budget")
                if budget is None:
                    mode = self.thinking_options.get("mode")
                    if mode == "dynamic":
                        budget = -1
                    elif mode == "disabled":
                        budget = 0
                    else:
                        budget = 8192

                config["thinking_config"] = self.genai_types.ThinkingConfig(
                    thinking_budget=int(budget)
                )
            except Exception as exc:
                print(f"[WARN] Gemini thinking_config unavailable: {exc}")

                       
        start = time.perf_counter()
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[content],
                config=config
            )
            raw = getattr(resp, "text", "") or ""
            parsed = getattr(resp, "parsed", None)
            if parsed is None:
                parsed = extract_json(raw)
            usage = getattr(resp, "usage_metadata", None)
            if usage:
                tokens_in = (getattr(usage, "prompt_token_count", None)
                             or getattr(usage, "input_token_count", None)
                             or getattr(usage, "input_tokens", None))
                tokens_out = (getattr(usage, "candidates_token_count", None)
                              or getattr(usage, "output_token_count", None)
                              or getattr(usage, "output_tokens", None))
            else:
                tokens_in = tokens_out = None
        except Exception as e:
            raw = f'__ERROR__: {type(e).__name__}: {e}'
            parsed = None
            tokens_in = tokens_out = None

        e2e = (time.perf_counter() - start) * 1000.0
        if isinstance(raw, str) and raw.startswith('__ERROR__'):
            print(f'[ERROR][Gemini] {raw}')
        meta = dict(ttft_ms=e2e, e2e_ms=e2e, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=None)
        return raw, parsed, meta


    
                                                                                 

               
                            

    
class FireworksProvider(ModelProvider):
    """
    Fireworks AI Provider: OpenAI-compatible API with JSON Schema support.
    Supports vision-language models like Qwen3-VL-235B.

    Base URL: https://api.fireworks.ai/inference/v1
    Model format: accounts/fireworks/models/{model_name}
    """
    provider_label = "Fireworks"
    default_base_url = "https://api.fireworks.ai/inference/v1"

    def __init__(self, model:str, api_key:str, base_url:str=None, **kwargs):
        super().__init__(model, api_key, **kwargs)
        if not self.api_key:
            raise RuntimeError(f"{self.provider_label}: Please configure api_key in secrets.yaml.")
        from openai import OpenAI
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": base_url or self.default_base_url,
            "timeout": self.timeout,
        }
        default_headers = kwargs.get("default_headers")
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        self.client = OpenAI(**client_kwargs)

    def infer(self, prompt:str, images:List[str], json_schema:Dict[str, Any],
              stream:bool=True, few_shot_examples:Optional[List]=None)->Tuple[str, Optional[Dict[str,Any]], Dict[str,Any]]:
        """
        OpenAI-compatible chat inference with JSON Schema support.

        Supports:
        - Vision input (up to 30 images, max 10MB total base64)
        - JSON Schema structured output
        - Few-shot examples
        """
                                               
        has_reasoning = "reasoning" in json_schema.get("properties", {})
        reasoning_note = REASONING_INSTRUCTION if has_reasoning else ""

        content = []

                                                 
        if few_shot_examples:
            for example_images, example_text in few_shot_examples:
                                    
                for img_path in example_images:
                    if os.path.exists(img_path):
                        mime = guess_mime(img_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{IMG_CACHE.get_b64(img_path)}",
                                "detail": "high"
                            }
                        })
                                         
                content.append({"type": "text", "text": example_text})
                                            
            content.append({"type": "text", "text": "\n\nNow solve this new problem:\n\n"})

                                 
        content.append({
            "type": "text",
            "text": (
                prompt +
                "\nReturn ONLY valid JSON per schema." +
                (f"\n\n{reasoning_note}" if reasoning_note else "")
            )
        })

                            
        for p in images:
            mime = guess_mime(p)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{IMG_CACHE.get_b64(p)}",
                    "detail": "high"
                }
            })

        msgs = [
            {"role": "system", "content": "Output ONLY a JSON object that strictly matches the schema."},
            {"role": "user", "content": content}
        ]

        if self.thinking_enabled:
            print(f"[WARN] {self.provider_label} provider: thinking flag is not explicitly supported; model may ignore.")

                                                  
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "captcha_response",
                "schema": json_schema,
                "strict": True                                  
            }
        }

        start = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=0,
                response_format=response_format,
            )
            raw = resp.choices[0].message.content or ""
            tokens_in = getattr(getattr(resp, "usage", None), "prompt_tokens", None)
            tokens_out = getattr(getattr(resp, "usage", None), "completion_tokens", None)
        except Exception as e:
            raw, tokens_in, tokens_out = f'__ERROR__: {type(e).__name__}: {e}', None, None

        e2e = (time.perf_counter() - start) * 1000.0
        parsed = extract_json(raw)
        meta = dict(ttft_ms=e2e, e2e_ms=e2e, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=None)
        return raw, parsed, meta


class OpenRouterProvider(FireworksProvider):
    """
    OpenRouter Provider: OpenAI-compatible API with JSON Schema support.

    Base URL: https://openrouter.ai/api/v1
    Model format: provider/model, e.g. qwen/qwen3-vl-235b-a22b-instruct
    """

    provider_label = "OpenRouter"
    default_base_url = "https://openrouter.ai/api/v1"

               
                 

    
def make_provider(name:str, model:str, secrets:dict, timeout_sec:float,
                     thinking_enabled: bool = False,
                     thinking_options: Optional[Dict[str, Any]] = None) -> ModelProvider:
    """
    Create concrete Provider instance by name and inject api_key/base_url from secrets.
    Args:
        name: Provider name (openai/anthropic/gemini/fireworks/openrouter).
        model: Model name.
        secrets: Config dict (returned by load_secrets).
        timeout_sec: Timeout in seconds.
    Returns:
        ModelProvider subclass instance.
    Raises:
        ValueError: Unknown provider.
    """
    name_l = name.lower()
    prov_cfg = (secrets.get("providers") or {}).get(name_l, {})
    api_key = prov_cfg.get("api_key")

    merged_thinking: Dict[str, Any] = {}
    prov_thinking = prov_cfg.get("thinking")
    if isinstance(prov_thinking, dict):
        merged_thinking.update(prov_thinking)
    if isinstance(thinking_options, dict):
        merged_thinking.update(thinking_options)
    enable_flag = thinking_enabled or bool(merged_thinking)

    common_kwargs = dict(
        timeout_sec=timeout_sec,
        thinking_enabled=enable_flag,
        thinking_options=merged_thinking
    )

    if name_l == "openai":
        return OpenAIProvider(model=model, api_key=api_key, **common_kwargs)
    if name_l == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key, **common_kwargs)
    if name_l == "gemini":
        return GeminiProvider(model=model, api_key=api_key, **common_kwargs)
    if name_l == "fireworks":
        return FireworksProvider(model=model, api_key=api_key, base_url=prov_cfg.get("base_url"), **common_kwargs)
    if name_l == "openrouter":
        default_headers = {}
        if prov_cfg.get("http_referer"):
            default_headers["HTTP-Referer"] = prov_cfg["http_referer"]
        if prov_cfg.get("x_title"):
            default_headers["X-Title"] = prov_cfg["x_title"]
        return OpenRouterProvider(
            model=model,
            api_key=api_key,
            base_url=prov_cfg.get("base_url"),
            default_headers=default_headers or None,
            **common_kwargs,
        )
    raise ValueError(f"Unknown provider: {name}. Supported providers: openai, anthropic, gemini, fireworks, openrouter")



               
              

    
@dataclass
class TaskItem:
    """
    Task entry:
    - type: Task type name (directory name)
    - puzzle_id: Problem filename or identifier
    - prompt: Text instruction (with overwrites/pre-post fixes applied)
    - images: Input image absolute path list (original images)
    - gt: Ground Truth (for evaluation)
    """
    type: str
    puzzle_id: str
    prompt: str
    images: List[str]
    gt: Dict[str, Any]


SUPPORTED_TYPES = {
            
    "Dice_Count",
    "Symbol_Count",
    "Geometry_Click",
    "Image_Matching",
    "Relation_Match",
    "Hole_Counting",
    "Patch_Select",
    "Place_Dot",
              
    "Bingo",
    "Click_Order",
    "Dart_Count",
    "Image_Recognition",
    "Misleading_Click",
    "Object_Match",
    "Path_Finder",
    "Pick_Area",
    "Select_Animal",
    "Select_Animal_Optimized",
    "Unusual_Detection",
    "Coordinates",
    "Connect_Icon",
    "Rotation_Match",
}

TASK_ALIASES = {"Connect_icon": "Connect_Icon"}
DATASET_DIR_ALIASES = {"Connect_Icon": "Connect_icon"}

TYPE_REQUIRE_PER_ITEM = {"Geometry_Click", "Image_Recognition", "Misleading_Click",
                         "Select_Animal","Select_Animal_Optimized","Patch_Select","Bingo", 
                         "Click_Order", "Connect_Icon", "Coordinates", "Dice_Count",
                        "Dart_Count", "Image_Matching", "Object_Match", "Pick_Area",
                        "Place_Dot", "Relation_Match", "Rotation_Match", "Symbol_Count",
                        "Unusual_Detection", "Path_Finder"}

                                                    
TYPE_IGNORE_PER_ITEM = {}
                                                   
                                                          


                        
                         
ID_REQUIRE_PER_ITEM = {
                                                           
}
                         
ID_IGNORE_PER_ITEM = {
                                             
}


def _is_git_lfs_pointer(text: str) -> bool:
    return text.strip().startswith("version https://git-lfs.github.com/spec/v1")


def load_ground_truth(type_dir: str) -> Dict[str, Any]:
    """
    Read ground_truth.json for a task type (reject LFS pointers).
    """
    gt_path = os.path.join(type_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"ground_truth.json does not exist: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        raw = f.read()
    if _is_git_lfs_pointer(raw):
        raise RuntimeError(f"{gt_path} is a Git LFS pointer. Please ensure you have pulled the actual JSON file.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse ground_truth.json: {gt_path} - {e}")



def _load_prompts_yaml(path: Optional[str]) -> dict:
    """Backward compatible: prompts.yaml can be old {type: str} format or new structured format."""
    if not path:
        return {}
    import yaml, io
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
                               
    if data and "types" not in data and "default" not in data and "by_id" not in data:
        data = {"version": 0, "types": data}
          
    data.setdefault("default", {})
    data.setdefault("types", {})
    data.setdefault("templates", {})
    data.setdefault("by_id", {})
    return data

def load_few_shot_examples(few_shot_file: str = "./few_shot_examples.yaml") -> Dict[str, List[Dict]]:
    """
    Load few-shot example configuration

    Args:
        few_shot_file: Few-shot config file path

    Returns:
        dict, key is task type, value is example list
    """
    if not few_shot_file or not os.path.exists(few_shot_file):
        return {}

    with open(few_shot_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config if config else {}


def _collect_image_paths(base_dir: pathlib.Path, name: str) -> List[str]:
    """
    Resolve image references for few-shot assets. Supports direct files and directories.
    """
    if not name:
        return []

    paths: List[str] = []
    targets = base_dir / name

    def add_from_path(path: pathlib.Path) -> None:
        if path.is_dir():
            for file in sorted(path.iterdir()):
                if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append(str(file))
        elif path.is_file():
            paths.append(str(path))

    if targets.exists():
        add_from_path(targets)
    else:
        stem = pathlib.Path(name).stem
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidate = base_dir / f"{stem}{ext}"
            if candidate.exists():
                add_from_path(candidate)
                break

    return paths

def build_few_shot_content(
    task_type: str,
    few_shot_examples: Dict,
    dataset_root: Optional[str] = None,
    few_shot_assets_root: Optional[pathlib.Path] = None,
    n_shot: Optional[int] = None
) -> List:
    """
    Build few-shot example content (image path list + concise answer text)

    Note: Answer data is now read from cognition.few_shot_answers (hardcoded),
    no longer depends on ground_truth.json to avoid matching issues from filename changes.

    Args:
        task_type: Task type
        few_shot_examples: Few-shot config dict (only used to specify which examples to use)
        dataset_root: Dataset root directory
        few_shot_assets_root: Few-shot assets root directory

    Returns:
        Example content list: [(images_list, answer_text), ...]
    """
                
    try:
        from .few_shot_answers import get_all_examples
    except ImportError:
        print("⚠️ Warning: Cannot import cognition.few_shot_answers, few-shot functionality unavailable")
        return []

    if task_type not in few_shot_examples:
        return []

    examples_data = few_shot_examples[task_type].get("examples", [])
    if isinstance(n_shot, int) and n_shot > 0:
        examples_data = examples_data[:n_shot]
    if not examples_data:
        return []

                         
    hardcoded_examples = get_all_examples(task_type)
    if not hardcoded_examples:
                                      
        print(f"⚠️ Warning: {task_type} has no hardcoded answer in few_shot_answers.py, falling back to YAML")
        hardcoded_examples = examples_data

             
    FEW_SHOT_ASSETS_ROOT = pathlib.Path(os.environ.get("FEW_SHOT_ASSETS_ROOT", "./few_shot_assets")).resolve()
    assets_root = pathlib.Path(few_shot_assets_root or FEW_SHOT_ASSETS_ROOT)
    task_dir_name = task_type.split('(')[0].strip()
    type_dir = assets_root / task_dir_name
    if not type_dir.exists() and dataset_root:
                            
        type_dir = pathlib.Path(dataset_root) / task_dir_name
    result = []

    for i, yaml_example in enumerate(examples_data, 1):
                       
        filename = yaml_example.get("filename")
        if not filename:
            continue

                          
        example = None
        for hc_ex in hardcoded_examples:
            if hc_ex.get("filename") == filename:
                example = hc_ex
                break

                                 
        if example is None:
            example = yaml_example

        images: List[str] = []
        seen: set[str] = set()

                                                 
                                                         
        puzzle_id_only_tasks = ["Dart_Count", "Connect_icon", "Coordinates", "Rotation_Match", "Object_Match"]

                                      
        if filename and task_type not in puzzle_id_only_tasks:
            for path in _collect_image_paths(type_dir, filename):
                if path not in seen:
                    images.append(path)
                    seen.add(path)

                        
                         
                                    
                                                                                                             
                                                              
                                         
        additional_image_fields = [
            "order_image",                        
            "reference_image",                                                                                                         
            "object_base_image",                     
            "component_image"                      
        ]

        for field in additional_image_fields:
            if field in example:
                img_value = example[field]
                if isinstance(img_value, str):
                    for path in _collect_image_paths(type_dir, img_value):
                        if path not in seen:
                            images.append(path)
                            seen.add(path)

                                             
                                            
                                                                              
        option_field = None
        if "options" in example:
            option_field = "options"
        elif "option_images" in example:
            option_field = "option_images"

        if option_field:
            option_list = example[option_field]
            if isinstance(option_list, list):
                for option_img in option_list:
                    if isinstance(option_img, str):
                        for path in _collect_image_paths(type_dir, option_img):
                            if path not in seen:
                                images.append(path)
                                seen.add(path)


        answer = example.get("answer")
        target_object = example.get("target_object")

        # Build the few-shot text with task description if available
        if answer is not None:
            if target_object and task_type == "Patch_Select":
                # For Patch_Select, include the specific task prompt
                task_prompt = f"Select all squares with {target_object}"
                answer_text = f"Few-shot example {i} (reference only, do NOT solve):\nTask: {task_prompt}\nAnswer JSON = {json.dumps(answer, ensure_ascii=False)}"
            else:
                answer_text = f"Few-shot example {i} (reference only, do NOT solve): Answer JSON = {json.dumps(answer, ensure_ascii=False)}"
        else:
            answer_text = f"Few-shot example {i} (reference only, do NOT solve): (no answer available)"

        if images:
            result.append((images, answer_text))

    return result

def _resolve_prompt_cfg(cfg: dict, task_type: str, puzzle_id: str) -> dict:
    """
    Parse configuration by type and problem ID:
    Priority: by_id override > types[task_type] > default
    Returns dict: {"mode": "merge|replace|auto|gt", "rules": str|None, "template": str|None, "override": str|None}
    """
    by_id = cfg.get("by_id", {})
                                                   
    override = by_id.get(f"{task_type}/{puzzle_id}") or\
               (isinstance(by_id.get(task_type), dict) and by_id[task_type].get(puzzle_id))

    tcfg = cfg.get("types", {}).get(task_type, {})
    dcfg = cfg.get("default", {})
                                                 
    if isinstance(tcfg, str):
        return {"mode": "replace", "rules": None, "template": None, "override": tcfg}

    mode = tcfg.get("mode") or dcfg.get("mode") or "merge"
    rules = tcfg.get("rules") or dcfg.get("rules")
    template = cfg.get("templates", {}).get(task_type) or cfg.get("templates", {}).get("default")

    return {"mode": mode, "rules": rules, "template": template, "override": override}

def _render_template(template: str, context: dict) -> str:
    """Minimal template: supports {{gt_prompt}}, {{type}}, {{pid}} three placeholders; avoids format brace conflicts."""
    if not template:
        return ""
    out = template.replace("{{gt_prompt}}", context.get("gt_prompt", ""))
    out = out.replace("{{type}}", context.get("type", ""))
    out = out.replace("{{pid}}", context.get("pid", ""))
    return out

def _choose_prompt(entry: dict,
                   task_type: str,
                   puzzle_id: str,
                   default_prompt: str,
                   prompts_cfg: dict,                                                
                   prefix: str,
                   suffix: str,
                   mode: str = "auto",
                   prompt_cfg: dict | None = None,
                   **_compat                                                
                   ) -> str:
    """
    Final prompt selection (supports gt/opt/merge/auto + structured prompts.yaml + manual strategy switch)
    - Compatible: old flat overrides ({Type: "text", "default": "text", "Type:ID": "text"})
    """
    cfg = prompt_cfg if prompt_cfg is not None else (prompts_cfg or {})

                    
    base_raw = (entry.get("prompt") or entry.get("question") or entry.get("instruction") or "").strip()

                                                  
    if mode == "gt":
        use_per_item = bool(base_raw)
        base = base_raw
    else:
        def _should_use_per_item(t: str, pid: str, raw: str) -> bool:
            if pid in ID_REQUIRE_PER_ITEM.get(t, set()): return True
            if pid in ID_IGNORE_PER_ITEM.get(t, set()): return False
            if t in TYPE_REQUIRE_PER_ITEM: return True
            if t in TYPE_IGNORE_PER_ITEM: return False
            return bool(raw)
        use_per_item = _should_use_per_item(task_type, puzzle_id, base_raw)
        base = base_raw if use_per_item else ""

                                                  
    resolved = _resolve_prompt_cfg(cfg, task_type, puzzle_id) if cfg else {}
    cfg_mode = (resolved.get("mode") or "merge")
    if cfg_mode == "replace":
        cfg_mode = "opt"
    rules_text = (resolved.get("rules") or "").strip() or None
    override_text = (resolved.get("override") or "").strip() or None
    template_text = (resolved.get("template") or "").strip() or None

                                         
    flat = _compat.get("overrides")
    flat_by_id = flat_type = flat_def = None
    if isinstance(flat, dict):
        flat_by_id = flat.get(f"{task_type}:{puzzle_id}") or flat.get(f"{task_type}/{puzzle_id}")
        flat_type  = flat.get(task_type)
        flat_def   = flat.get("default")

             
    if mode in ("gt","merge","opt"):
        effective_mode = mode
    else:        
        effective_mode = "merge" if use_per_item else "opt"

             
    if effective_mode == "gt":
        p = base if base else default_prompt

    elif effective_mode == "opt":
                                                                                              
        if override_text:
            p = override_text
        elif flat_by_id or flat_type or flat_def:
            p = flat_by_id or flat_type or flat_def
        elif template_text:
            p = _render_template(template_text, {"gt_prompt": base or default_prompt, "type": task_type, "pid": puzzle_id})
        elif rules_text:
            p = rules_text
        else:
            p = base or default_prompt

    else:         
        core = base or default_prompt
        extras = []
        if template_text:
            extras.append(_render_template(template_text, {"gt_prompt": core, "type": task_type, "pid": puzzle_id}))
        if override_text:
            extras.append(override_text)
        if rules_text:
            extras.append(rules_text)
                                  
        for x in (flat_by_id, flat_type, flat_def):
            if x: extras.append(x)
        p = core + (("\n\n" + "\n\n".join(extras)) if extras else "")

            
    if prefix:
        p = f"{prefix}\n{p}"
    if suffix:
        p = f"{p}\n{suffix}"
    return p





def build_tasks(
    dataset_root: str,
    types: List[str],
    max_per_type: int = None,
    prompts_cfg: dict | None = None,
    prompt_prefix: str = "",
    prompt_suffix: str = "",
    prompt_mode: str = "auto",                              
    exclude_examples: Optional[Dict[str, List[str]]] = None
) -> List[TaskItem]:
    prompts_cfg = prompts_cfg or {}
    exclude_examples = exclude_examples or {}
    tasks: List[TaskItem] = []
    for requested_type in types:
        t = TASK_ALIASES.get(requested_type, requested_type)
        if t not in SUPPORTED_TYPES:
            print(f"[SKIP] Type not yet supported: {requested_type}")
            continue
        type_dir = os.path.join(dataset_root, DATASET_DIR_ALIASES.get(t, t))
        try:
            gt = load_ground_truth(type_dir)
        except Exception as e:
            print(f"[SKIP] Failed to read GT {requested_type}: {e}")
            continue

        puzzle_ids = list(gt.keys())

                              
        excluded_files = exclude_examples.get(t, [])
        if excluded_files:
            excluded_set = set(excluded_files)
            excluded_stems = {os.path.splitext(name)[0] for name in excluded_files}
            puzzle_ids = [
                pid for pid in puzzle_ids
                if pid not in excluded_set and os.path.splitext(pid)[0] not in excluded_stems
            ]

                                                            
                               
        if isinstance(max_per_type, int) and max_per_type > 0 and len(puzzle_ids) > max_per_type:
            puzzle_ids = random.sample(puzzle_ids, k=max_per_type)
        else:
            random.shuffle(puzzle_ids)

        for pid in puzzle_ids:
            entry = gt[pid]

                           
            if t == "Dice_Count":
                default_prompt = ("Count ONLY the pips visible on the TOP faces of all dice. "
                                  "Do NOT count side/hidden faces or reflections. "
                                  "Return JSON {\"answer_type\":\"number\",\"value\":N}.")
                prompt = _choose_prompt(entry, t, pid, default_prompt, prompts_cfg or {}, prompt_prefix, prompt_suffix, mode=prompt_mode)
                img = os.path.join(type_dir, pid)
                if not os.path.isfile(img): 
                    print(f"[SKIP] File does not exist: {img}"); 
                    continue
                tasks.append(TaskItem(t, pid, prompt, [img], {"sum": entry.get("sum")}))

            elif t == "Symbol_Count":
                default_prompt = (
                    "Count the target symbols in the static image. "
                    "Return JSON {\"answer_type\":\"number\",\"value\":N}."
                )
                prompt = _choose_prompt(
                    entry,
                    t,
                    pid,
                    default_prompt,
                    prompts_cfg or {},
                    prompt_prefix,
                    prompt_suffix,
                    mode=prompt_mode,
                )
                image_name = entry.get("image") or pid
                img = os.path.join(type_dir, image_name)
                if not os.path.isfile(img):
                    img = _stem_glob(type_dir, pid) or img
                if not os.path.isfile(img):
                    print(f"[SKIP] File does not exist: {img}")
                    continue
                count = entry.get("count", entry.get("sum"))
                if count is None:
                    print(f"[SKIP] {t}/{pid} missing count")
                    continue
                tasks.append(TaskItem(t, pid, prompt, [img], {"count": int(count)}))

            elif t == "Geometry_Click":
                targets_type = str((entry.get("answer") or {}).get("type") or "").strip()
                if targets_type.lower().startswith("letter "):
                    obj = targets_type.split(" ", 1)[1].strip() or "targets letter"
                    semantic = f"Click the CENTER of letter '{obj}'."
                elif targets_type:
                    semantic = f"Click the CENTER of the targets {targets_type}."
                else:
                    semantic = "Click the CENTER of the specified geometric targets."

                default_prompt = (
                    f"{semantic}\n"
                    "Return ONLY JSON {\"answer_type\":\"single_point\",\"point\":{\"x\":...,\"y\":...}} (pixels)."
                )

                                                                 
                prompt = _choose_prompt(
                    entry=entry,                                              
                    task_type=t,                                   
                    puzzle_id=pid,                                           
                    default_prompt=default_prompt,
                    prompts_cfg=prompts_cfg or {},       
                    prefix=prompt_prefix,
                    suffix=prompt_suffix,
                    mode=prompt_mode
                )

                             
                img = os.path.join(type_dir, pid)
                if not os.path.isfile(img):
                    print(f"[SKIP] File does not exist: {img}")
                    continue

                                                                                       
                ans = entry.get("answer") or {}
                area = ans.get("area")
                bbox = None
                if (isinstance(area, (list, tuple)) and len(area) == 2
                    and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in area)):
                    (x0, y0), (x1, y1) = area
                    x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
                    y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)
                    bbox = [float(x0), float(y0), float(x1), float(y1)]
                else:
                    print(f"[SKIP] {t}/{pid} illegal area format: {area}")
                    continue

                                                  
                shape = targets_type or "region"
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                tol = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2.0
                tasks.append(
                    TaskItem(
                        t,
                        pid,
                        prompt,
                        [img],
                        {
                            "bbox": bbox,
                            "shape": shape,
                            "targets_position": {"x": cx, "y": cy},
                            "tolerance": tol,
                        },
                    )
                )


            elif t == "Image_Matching":
                ref = entry.get("reference_image")
                                               
                options = entry.get("option_images") or entry.get("options", [])
                if not ref or not options:
                    print(f"[SKIP] {t}/{pid} Missing reference/option_images")
                    continue
                default_prompt = (
                    "The FIRST image is the REFERENCE, followed by OPTION images indexed 0..N-1 in order. "
                    "Return JSON {\"answer_type\":\"classify\",\"index\":k} with 0-based index only."
                )
                prompt = _choose_prompt(entry, t, pid, default_prompt, prompts_cfg or {}, prompt_prefix, prompt_suffix, mode=prompt_mode)
                
                img_ref = os.path.join(type_dir, ref)
                imgs_opt = [os.path.join(type_dir, p) for p in options]
                ok = True
                for p in [img_ref]+imgs_opt:
                    if not os.path.isfile(p): 
                        print(f"[SKIP] File does not exist: {p}"); 
                        ok=False; break
                if not ok: 
                    continue
                tasks.append(TaskItem(t, pid, prompt, [img_ref] + imgs_opt,
                                      {"correct_index": entry.get("correct_option_index", entry.get("correct_index", 0)),
                                       "num_options": len(options)}))

            elif t == "Relation_Match":
                ref = entry.get("reference_image")
                options = entry.get("option_images") or entry.get("options", [])
                if not ref or not options:
                    print(f"[SKIP] {t}/{pid} Missing reference/option_images")
                    continue
                default_prompt = (
                    "The FIRST image is the REFERENCE, followed by OPTION images indexed 0..N-1 in order. "
                    "Choose the option matching the reference relation. "
                    "Return JSON {\"answer_type\":\"classify\",\"index\":k} with 0-based index only."
                )
                prompt = _choose_prompt(
                    entry,
                    t,
                    pid,
                    default_prompt,
                    prompts_cfg or {},
                    prompt_prefix,
                    prompt_suffix,
                    mode=prompt_mode,
                )

                img_ref = os.path.join(type_dir, ref)
                imgs_opt = [os.path.join(type_dir, p) for p in options]
                ok = True
                for p in [img_ref] + imgs_opt:
                    if not os.path.isfile(p):
                        print(f"[SKIP] File does not exist: {p}")
                        ok = False
                        break
                if not ok:
                    continue
                tasks.append(
                    TaskItem(
                        t,
                        pid,
                        prompt,
                        [img_ref] + imgs_opt,
                        {
                            "correct_index": int(entry.get("correct_index", 0)),
                            "num_options": len(options),
                        },
                    )
                )

            elif t in ("Patch_Select", "Hole_Counting"):
                               
                img = os.path.join(type_dir, pid)
                if not os.path.isfile(img):
                    print(f"[SKIP] File does not exist: {img}")
                    continue

                                     
                if t == "Hole_Counting":
                    target_holes = entry.get("target_holes")
                    targets = _get_first(
                        entry,
                        "target_object",
                        "targets_object",
                        default=f"tiles with exactly {target_holes} holes",
                    )
                else:
                    targets = _get_first(entry, "target_object", "targets_object", default="the target object")
                gs = entry.get("grid_size", [5, 5])
                if not (isinstance(gs, (list, tuple)) and len(gs) == 2):
                    gs = [5, 5]
                R, C = int(gs[0]), int(gs[1])

                                                        
                default_prompt = (
                    f"The image already shows a {R}x{C} grid.\n"
                    f"Select ALL cells that contain the targets: {targets}.\n"
                    f"Indexing is 0-based and row-major: cell_index = r*{C} + c.\n"
                    "Return JSON {\"answer_type\":\"multi_select\",\"indices\":[...]} "
                    "with unique, sorted indices only. No explanations."
                )

                                                                 
                prompt = _choose_prompt(
                    entry=entry,
                    task_type=t,
                    puzzle_id=pid,
                    default_prompt=default_prompt,
                    prompts_cfg=prompts_cfg or {},
                    prefix=prompt_prefix,
                    suffix=prompt_suffix,
                    mode=prompt_mode
                )

                                                               
                indices_gt = entry.get("correct_patches")
                if indices_gt is None:
                    indices_gt = entry.get("correct_selections", [])
                if not isinstance(indices_gt, list):
                    indices_gt = []

                                            
                tasks.append(
                    TaskItem(
                        t, pid, prompt, [img],
                        {"grid_size": [R, C], "correct_patches": indices_gt}
                    )
                )


            elif t == "Place_Dot":
                                              
                default_prompt = (
                    "Place a dot at the targets position and return the CENTER point "
                    "as JSON {\"answer_type\":\"single_point\",\"point\":{\"x\":..,\"y\":..}} in pixels."
                )
                prompt = _choose_prompt(entry, t, pid, default_prompt,
                                        prompts_cfg or {}, prompt_prefix, prompt_suffix,
                                        mode=prompt_mode)

                img = os.path.join(type_dir, pid)
                if not os.path.isfile(img):
                    print(f"[SKIP] File does not exist: {img}")
                    continue

                tp = entry.get("targets_position") or entry.get("target_position")
                if isinstance(tp, dict) and "x" in tp and "y" in tp:
                    gx, gy = float(tp["x"]), float(tp["y"])
                    tp_norm = {"x": gx, "y": gy}
                elif isinstance(tp, (list, tuple)) and len(tp) == 2:
                    gx, gy = float(tp[0]), float(tp[1])
                    tp_norm = {"x": gx, "y": gy}
                else:
                    print(f"[SKIP] {t}/{pid} illegal targets_position: {tp!r}")
                    continue

                tol = float(entry.get("tolerance", 15.0))           

                                          
                tasks.append(
                    TaskItem(t, pid, prompt, [img], {
                        "targets_position": tp_norm,                        
                        "tolerance": tol
                    })
                )


                               
            elif t in ("Select_Animal","Unusual_Detection"):
                                            
                default_prompt = (
                    "Select ALL grid cells (0-based, row-major) that satisfy the condition. "
                    "Return JSON {\"answer_type\":\"multi_select\",\"indices\":[...]}."
                )
                prompt = _choose_prompt(entry, t, pid, default_prompt, prompts_cfg or {}, prompt_prefix, prompt_suffix, mode=prompt_mode)
                img = os.path.join(type_dir, pid)
                if not os.path.isfile(img):
                    print(f"[SKIP] File does not exist: {img}")
                    continue
                         
                indices_gt = _get_first(entry, "correct_selections", "correct_patches", "answer", default=[])
                tasks.append(TaskItem(t, pid, prompt, [img], {"indices_gt": indices_gt}))

            elif t == "Select_Animal_Optimized":
                default_prompt = (
                    "Click on all of the target's eyes and return JSON "
                    "{\"answer_type\":\"multi_point\",\"points\":[{\"x\":..,\"y\":..}, ...]}. "
                    "Include one point per eye; order does not matter."
                )
                prompt = _choose_prompt(entry, t, pid, default_prompt, prompts_cfg or {}, prompt_prefix, prompt_suffix, mode=prompt_mode)
                img = os.path.join(type_dir, pid)
                if not os.path.isfile(img):
                    print(f"[SKIP] File does not exist: {img}")
                    continue
                raw_positions = _get_first(entry, "target_position", "targets_position", "answer", default=[])
                gt_pts = _normalize_points(raw_positions)
                if not gt_pts:
                    print(f"[SKIP] {t}/{pid} missing target_position list")
                    continue
                tol = float(entry.get("tolerance", 15.0))
                tasks.append(TaskItem(t, pid, prompt, [img], {"targets_positions": gt_pts, "tolerance": tol}))

            elif t == "Path_Finder":
                                                         
                default_prompt_cls = (
                    "Choose the option depicting the same physical location as the reference."
                    " Return JSON {\"answer_type\":\"classify\",\"index\":k} (0-based)."
                )
                default_prompt_multi = (
                    "Select ALL grid cells (0-based, row-major) that match the reference view."
                    " Return JSON {\"answer_type\":\"multi_select\",\"indices\":[...]}"
                )
                                               
                has_ref_opts = bool(entry.get("reference_image")) and bool(entry.get("option_images") or entry.get("options"))

                default_prompt = default_prompt_cls if has_ref_opts else default_prompt_multi
                prompt = _choose_prompt(entry, t, pid, default_prompt,
                                        prompts_cfg or {}, prompt_prefix, prompt_suffix,
                                        mode=prompt_mode)

                if has_ref_opts:
                    ref = os.path.join(type_dir, entry["reference_image"])
                                                   
                    option_list = entry.get("option_images") or entry.get("options", [])
                    opts = [os.path.join(type_dir, p) for p in option_list]
                    if not (os.path.isfile(ref) and all(os.path.isfile(p) for p in opts)):
                        print(f"[SKIP] {t}/{pid} missing image files")
                        continue
                                                                                    
                    corr = int(_get_first(entry, "correct_option","correct_option_index","correct_index","answer", default=0))
                    tasks.append(TaskItem(t, pid, prompt, [ref]+opts, {"correct_index": corr}))
                else:
                    img = os.path.join(type_dir, pid)
                    if not os.path.isfile(img):
                        print(f"[SKIP] File does not exist: {img}")
                        continue
                    indices_gt = _get_first(entry, "correct_selections", "correct_patches", "answer", default=[])
                    tasks.append(TaskItem(t, pid, prompt, [img], {"indices_gt": indices_gt}))


            elif t in ("Dart_Count","Coordinates","Object_Match"):
                                                 
                default_prompt = (
                    "Choose the correct option and return JSON {\"answer_type\":\"classify\",\"index\":k} (0-based)."
                )
                prompt = _choose_prompt(entry, t, pid, default_prompt, prompts_cfg or {}, prompt_prefix, prompt_suffix, mode=prompt_mode)
                                                    
                ref = entry.get("reference_image")
                                               
                options = entry.get("option_images") or entry.get("options", [])
                imgs: List[str] = []
                if ref and options:
                    img_ref = os.path.join(type_dir, ref)
                    imgs_opt = [os.path.join(type_dir, p) for p in options]
                    ok = os.path.isfile(img_ref) and all(os.path.isfile(p) for p in imgs_opt)
                    if not ok:
                        print(f"[SKIP] {t}/{pid} missing image files")
                        continue
                    imgs = [img_ref] + imgs_opt
                else:
                    single = os.path.join(type_dir, pid)
                    if not os.path.isfile(single):
                        print(f"[SKIP] File does not exist: {single}")
                        continue
                    imgs = [single]
                correct_idx = _get_first(entry, "correct_option_index","correct_index","answer", default=0)
                tasks.append(TaskItem(t, pid, prompt, imgs, {"correct_index": int(correct_idx)}))

            elif t == "Pick_Area":
                default_prompt = (
                    "Return a single point (pixel coordinates) that lies INSIDE the targets area. "
                    "JSON only: {\"answer_type\":\"single_point\",\"point\":{\"x\":..,\"y\":..}}."
                )
                prompt = _choose_prompt(entry, t, pid, default_prompt, prompts_cfg or {}, prompt_prefix, prompt_suffix, mode=prompt_mode)
                img = os.path.join(type_dir, pid)
                if not os.path.isfile(img):
                    print(f"[SKIP] File does not exist: {img}")
                    continue
                area = _get_first(entry, "area", "answer", default=None)
                if isinstance(area, dict) and "area" in area:
                    area = area["area"]
                if area is None:
                    print(f"[SKIP] {t}/{pid} missing area")
                    continue
                tasks.append(TaskItem(t, pid, prompt, [img], {"area_box": area}))

            elif t == "Click_Order":
                default_prompt = (
                    "If provided, use the reference image to follow the required sequence. "
                    "Click the targetss IN ORDER and return JSON "
                    "{\"answer_type\":\"click_order\",\"points\":[{\"x\":..,\"y\":..}, ...]} (pixels)."
                )
                prompt = _choose_prompt(entry, t, pid, default_prompt, prompts_cfg or {}, prompt_prefix, prompt_suffix, mode=prompt_mode)
                puzzle_img = os.path.join(type_dir, pid)
                if not os.path.isfile(puzzle_img):
                    print(f"[SKIP] File does not exist: {puzzle_img}")
                    continue
                images: List[str] = []
                order_img_name = entry.get("order_image") or entry.get("reference_image") or entry.get("order")
                if order_img_name:
                    order_img_path = os.path.join(type_dir, order_img_name)
                    if os.path.isfile(order_img_path):
                        images.append(order_img_path)
                    else:
                        print(f"[Warning] {t}/{pid} missing order_image: {order_img_path}")
                images.append(puzzle_img)
                pts = _get_first(entry, "answer", "points", default=[])
                tol = float(_get_first(entry, "tolerance", "tol_px", default=40.0))
                                             
                points_gt = []
                for p in pts:
                    if isinstance(p, dict) and "x" in p and "y" in p:
                        points_gt.append({"x":float(p["x"]), "y":float(p["y"])})
                    elif isinstance(p, (list,tuple)) and len(p)==2:
                        points_gt.append({"x":float(p[0]), "y":float(p[1])})
                if not points_gt:
                    print(f"[SKIP] {t}/{pid} missing point list")
                    continue
                tasks.append(TaskItem(t, pid, prompt, images, {"points_gt": points_gt, "tolerance": tol}))

            elif t == "Bingo":
                default_prompt = (
                    "Return exactly two cell indices to swap as JSON "
                    "{\"answer_type\":\"swap\",\"correct\":[i,j]} (0-based)."
                )
                prompt = _choose_prompt(entry, t, pid, default_prompt, prompts_cfg or {}, prompt_prefix, prompt_suffix, mode=prompt_mode)
                img = os.path.join(type_dir, pid)
                if not os.path.isfile(img):
                    print(f"[SKIP] File does not exist: {img}")
                    continue
                                
                swap_corrects = _get_first(entry, "answer", "swap_corrects", default=[])
                                   
                norm_corrects = []
                for p in swap_corrects:
                    if isinstance(p, (list,tuple)) and len(p)==2:
                        norm_corrects.append([int(p[0]), int(p[1])])
                    elif isinstance(p, dict) and "i" in p and "j" in p:
                        norm_corrects.append([int(p["i"]), int(p["j"])])
                if not norm_corrects:
                    print(f"[SKIP] {t}/{pid} missing swap_corrects")
                    continue
                tasks.append(TaskItem(t, pid, prompt, [img], {"swap_corrects": norm_corrects}))


            elif t == "Misleading_Click":
                default_prompt = (
                    "Return a SAFE click point INSIDE the image and OUTSIDE the forbidden area. "
                    "JSON only: {\"answer_type\":\"single_point\",\"point\":{\"x\":..,\"y\":..}}."
                )
                prompt = _choose_prompt(entry, t, pid, default_prompt, prompts_cfg or {}, prompt_prefix, prompt_suffix, mode=prompt_mode)
                img = os.path.join(type_dir, pid)
                if not os.path.isfile(img):
                    print(f"[SKIP] File does not exist: {img}")
                    continue
                avoid = _get_first(entry, "avoid_area","forbid_area","answer", default=None)
                                     
                try:
                    from PIL import Image
                    with Image.open(img) as im:
                        w,h = im.size
                except Exception:
                    print(f"[SKIP] {t}/{pid} failed to read dimensions")
                    continue
                tasks.append(TaskItem(t, pid, prompt, [img], {"avoid_area": avoid, "image_size": (w,h)}))

            elif t == "Image_Recognition":
                default_prompt = (
                    "Select ALL grid cells (0-based, row-major) that satisfy the condition. "
                    "Return JSON {\"answer_type\":\"multi_select\",\"indices\":[...]}."
                )
                prompt = _choose_prompt(entry, t, pid, default_prompt, prompts_cfg or {}, prompt_prefix, prompt_suffix, mode=prompt_mode)

                                                                 
                sub = entry.get("subfolder") or entry.get("images_dir") or ""
                base_dir = os.path.join(type_dir, sub) if sub else type_dir

                names = entry.get("images") or []                              
                if names:
                    imgs = [os.path.join(base_dir, n) for n in names]
                else:
                                                           
                    imgs = _list_images_in_dir(base_dir)

                                          
                imgs = [p for p in imgs if os.path.isfile(p)]
                if len(imgs) < 9:
                    print(f"[SKIP] {t}/{pid} missing image files: parsed to {len(imgs)} images (need ≥9)")
                    continue
                imgs = imgs[:9]               

                indices_gt = _get_first(entry, "correct_selections", "correct_patches", "answer", default=[])
                tasks.append(TaskItem(t, pid, prompt, imgs, {"indices_gt": indices_gt}))


            elif t == "Connect_Icon":
                default_prompt = (
                    "Choose the correct option and return JSON {\"answer_type\":\"classify\",\"index\":k} (0-based)."
                )
                prompt = _choose_prompt(entry, t, pid, default_prompt, prompts_cfg or {}, prompt_prefix, prompt_suffix, mode=prompt_mode)

                                                     
                ref_name = entry.get("reference_image") or entry.get("reference")
                opt_names = entry.get("options") or entry.get("option_images") or entry.get("candidates")

                if not (ref_name and isinstance(opt_names, list) and opt_names):
                    print(f"[SKIP] {t}/{pid} missing reference_image/options")
                    continue

                img_ref = os.path.join(type_dir, ref_name)
                imgs_opt = [os.path.join(type_dir, n) for n in opt_names]

                if not (os.path.isfile(img_ref) and all(os.path.isfile(p) for p in imgs_opt)):
                    print(f"[SKIP] {t}/{pid} missing image files (reference or options do not exist)")
                    continue

                correct_idx = int(_get_first(entry, "correct_option", "correct_option_index", "answer", default=0))
                tasks.append(TaskItem(t, pid, prompt, [img_ref] + imgs_opt, {"correct_index": correct_idx}))


            elif t == "Rotation_Match":
                default_prompt = (
                    "Return the rotation angle (degrees) that best aligns the object. "
                    "JSON only: {\"answer_type\":\"rotation\",\"angle\":...}."
                )
                prompt = _choose_prompt(entry, t, pid, default_prompt, prompts_cfg or {}, prompt_prefix, prompt_suffix, mode=prompt_mode)

                ref_name = entry.get("reference_image") or entry.get("reference")
                obj_name = entry.get("object_base_image") or entry.get("object_image") or entry.get("image") or entry.get("img")

                if not (ref_name and obj_name):
                    print(f"[SKIP] {t}/{pid} missing reference_image/object_base_image")
                    continue

                img_ref = os.path.join(type_dir, ref_name)                      
                img_obj = os.path.join(type_dir, obj_name)                                     

                if not (os.path.isfile(img_ref) and os.path.isfile(img_obj)):
                    print(f"[SKIP] {t}/{pid} missing image files (reference or object do not exist)")
                    continue

                                 
                ang = float(_get_first(entry, "correct_angle", "answer", "targets_angle_deg", default=0.0))
                ang_tol = float(_get_first(entry, "angle_tol_deg", "tolerance_deg", default=5.0))

                                              
                tasks.append(TaskItem(t, pid, prompt, [img_ref, img_obj],
                                    {"correct_angle": ang, "angle_tol_deg": ang_tol}))


            
            
            else:
                         
                print(f"[SKIP] no build logic yet: {t}/{pid}")
                continue

    return tasks



               
                             

    

def _clean_indices(xs):
    """Normalize indices to ascending, deduplicated int list; returns [] when passed None/empty."""
    out = []
    for v in (xs or []):
        try:
            out.append(int(v))
        except Exception:
            continue
    return sorted(set(out))


def evaluate_pass1(task:TaskItem, parsed:Optional[Dict[str,Any]])->bool:
    if parsed is None:
        return False
    t = task.type
    gt = task.gt
    try:
                        
        if t == "Dice_Count":
            return parsed.get("answer_type")=="number" and int(parsed.get("value")) == int(gt["sum"])

        if t == "Symbol_Count":
            return parsed.get("answer_type")=="number" and int(parsed.get("value")) == int(gt["count"])
        
        if task.type == "Geometry_Click":
            try:
                x = float(parsed["point"]["x"]); y = float(parsed["point"]["y"])
            except Exception:
                return False
            bbox = gt.get("bbox")
            if not (isinstance(bbox, (list,tuple)) and len(bbox)==4):
                return False
            x0,y0,x1,y1 = map(float, bbox)
            return (x0 <= x <= x1) and (y0 <= y <= y1)

        if t == "Image_Matching":
            return parsed.get("answer_type")=="classify" and int(parsed.get("index")) == int(gt["correct_index"])

        if t == "Relation_Match":
            return parsed.get("answer_type")=="classify" and int(parsed.get("index")) == int(gt["correct_index"])
        
        if t in ("Patch_Select", "Hole_Counting"):
            pred = sorted(set(int(i) for i in parsed.get("indices", [])))
            gold = sorted(set(int(i) for i in gt.get("correct_patches",[])))
            return pred == gold
        
        if task.type == "Place_Dot":
                                                                            
            try:
                pt = parsed["point"]
                x_pred, y_pred = float(pt["x"]), float(pt["y"])
            except Exception:
                return False

                                                              
            tp = gt.get("targets_position")
            tol = float(gt.get("tolerance", 15.0))

            gx = gy = None
            if isinstance(tp, dict) and "x" in tp and "y" in tp:
                gx, gy = float(tp["x"]), float(tp["y"])
            elif isinstance(tp, (list, tuple)) and len(tp) == 2:
                                     
                gx, gy = float(tp[0]), float(tp[1])

            if gx is None or gy is None:
                return False

                           
            dx, dy = x_pred - gx, y_pred - gy
            return (dx*dx + dy*dy) <= (tol * tol)

                         
        if task.type == "Path_Finder":
            return (
                parsed.get("answer_type") == "classify"
                and int(parsed.get("index")) == int(task.gt.get("correct_index"))
            )

        if t in ("Image_Recognition","Select_Animal","Unusual_Detection"):
            pred = _clean_indices(parsed.get("indices", []))
            gold = _clean_indices(gt.get("indices_gt", []))
            return pred == gold

        if t == "Select_Animal_Optimized":
            if parsed.get("answer_type") != "multi_point":
                return False
            pred_pts = _normalize_points(parsed.get("points"))
            gt_pts = _normalize_points(gt.get("targets_positions"))
            if not pred_pts or not gt_pts:
                return False
            tol = float(gt.get("tolerance", 15.0))
            matched, missing_gt, extra_pred = _match_points(pred_pts, gt_pts, tol)
            return matched == len(gt_pts) and missing_gt == 0 and extra_pred == 0

        if t in ("Dart_Count","Coordinates","Connect_Icon","Object_Match"):
            return parsed.get("answer_type")=="classify" and int(parsed.get("index")) == int(gt["correct_index"])

        if t == "Pick_Area":
            pt = parsed.get("point") or {}
            x,y = float(pt.get("x")), float(pt.get("y"))
            area = gt.get("area_box")                                 
            return _is_rect_hit(x,y,area)

        if t == "Click_Order":
            pts_pred = list(parsed.get("points") or [])
            pts_gt   = list(gt.get("points_gt") or [])
            if len(pts_pred) != len(pts_gt) or len(pts_gt) == 0:
                return False
            tol = float(gt.get("tolerance", 40.0))
            for p,q in zip(pts_pred, pts_gt):
                if not (("x" in p and "y" in p) and ("x" in q and "y" in q)):
                    return False
                if _point_dist(p,q) > tol:
                    return False
            return True

        if t == "Bingo":
            correct = parsed.get("correct") or []
            if not (isinstance(correct, (list,tuple)) and len(correct)==2):
                return False
            pred = _correct_as_set(correct)
            gold_corrects = [ _correct_as_set(p) for p in (gt.get("swap_corrects") or []) ]
            return pred in gold_corrects

        if t == "Rotation_Match":
            if parsed.get("answer_type")!="rotation":
                return False
            ang_pred = float(parsed.get("angle"))
            ang_gt   = float(gt.get("correct_angle"))
            tol = float(gt.get("angle_tol_deg", 5.0))
                                    
            diff = abs(((ang_pred - ang_gt + 180) % 360) - 180)
            return diff <= tol

        if t == "Misleading_Click":
                    
            pt = parsed.get("point") or {}
            x,y = float(pt.get("x")), float(pt.get("y"))
            w,h = gt.get("image_size",(None,None))
            if w is None or h is None:
                return False                         
            if not (0 <= x < w and 0 <= y < h):
                return False
                      
            avoid = gt.get("avoid_area")
            if avoid:
                if _is_rect_hit(x,y,avoid):
                    return False
            return True
    except Exception:
        return False
    return False


def _with_reasoning(schema: Dict[str, Any], *, include_reasoning: bool) -> Dict[str, Any]:
    """Append optional reasoning field (detailed reasoning process) to existing schema."""
    try:
        if include_reasoning:
            props = schema.setdefault("properties", {})
            if "reasoning" not in props:
                                          
                props["reasoning"] = {"type": "string", "maxLength": 8192}
        return schema
    except Exception:
        return schema


def build_json_schema(task_type:str, *, include_reasoning: bool = False)->Dict[str,Any]:
                    
    if task_type == "Dice_Count":
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["number"]},
                                              "value":{"type":"integer"}},
                "required":["answer_type","value"]}, include_reasoning=include_reasoning)
    if task_type == "Symbol_Count":
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["number"]},
                                              "value":{"type":"integer"}},
                "required":["answer_type","value"]}, include_reasoning=include_reasoning)
    if task_type in ("Geometry_Click","Place_Dot"):
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["single_point"]},
                                              "point":{"type":"object",
                                                       "properties":{"x":{"type":"number"},"y":{"type":"number"}},
                                                       "required":["x","y"]}},
                "required":["answer_type","point"]}, include_reasoning=include_reasoning)
    if task_type == "Image_Matching":
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["classify"]},
                                              "index":{"type":"integer"}},
                "required":["answer_type","index"]}, include_reasoning=include_reasoning)
    if task_type == "Relation_Match":
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["classify"]},
                                              "index":{"type":"integer"}},
                "required":["answer_type","index"]}, include_reasoning=include_reasoning)
    if task_type == "Patch_Select":
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["multi_select"]},
                                              "indices":{"type":"array","items":{"type":"integer"}}},
                "required":["answer_type","indices"]}, include_reasoning=include_reasoning)
    if task_type == "Hole_Counting":
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["multi_select"]},
                                              "indices":{"type":"array","items":{"type":"integer"}}},
                "required":["answer_type","indices"]}, include_reasoning=include_reasoning)

                    
    if task_type in ("Image_Recognition","Select_Animal","Unusual_Detection"):
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["multi_select"]},
                                              "indices":{"type":"array","items":{"type":"integer"}}},
                "required":["answer_type","indices"]}, include_reasoning=include_reasoning)

    if task_type == "Select_Animal_Optimized":
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["multi_point"]},
                                              "points":{"type":"array","items":{"type":"object",
                                                                               "properties":{"x":{"type":"number"},"y":{"type":"number"}},
                                                                               "required":["x","y"]}}},
                "required":["answer_type","points"]}, include_reasoning=include_reasoning)

    if task_type in ("Dart_Count","Coordinates","Connect_Icon","Object_Match","Path_Finder"):
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["classify"]},
                                              "index":{"type":"integer"}},
                "required":["answer_type","index"]}, include_reasoning=include_reasoning)

    if task_type == "Pick_Area":
                      
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["single_point"]},
                                              "point":{"type":"object",
                                                       "properties":{"x":{"type":"number"},"y":{"type":"number"}},
                                                       "required":["x","y"]}},
                "required":["answer_type","point"]}, include_reasoning=include_reasoning)

    if task_type == "Click_Order":
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["click_order"]},
                                              "points":{"type":"array",
                                                        "items":{"type":"object",
                                                                 "properties":{"x":{"type":"number"},"y":{"type":"number"}},
                                                                 "required":["x","y"]}}},
                "required":["answer_type","points"]}, include_reasoning=include_reasoning)

    if task_type == "Bingo":
                             
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["swap"]},
                                              "correct":{"type":"array","minItems":2,"maxItems":2,
                                                      "items":{"type":"integer"}}},
                "required":["answer_type","correct"]}, include_reasoning=include_reasoning)

    if task_type == "Rotation_Match":
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["rotation"]},
                                              "angle":{"type":"number"}},
                "required":["answer_type","angle"]}, include_reasoning=include_reasoning)

    if task_type == "Misleading_Click":
        return _with_reasoning({"type":"object","properties":{"answer_type":{"type":"string","enum":["single_point"]},
                                              "point":{"type":"object",
                                                       "properties":{"x":{"type":"number"},"y":{"type":"number"}},
                                                       "required":["x","y"]}},
                "required":["answer_type","point"]}, include_reasoning=include_reasoning)

              
    return _with_reasoning({"type":"object"}, include_reasoning=include_reasoning)


def _revision_dataset_summary(dataset_root: str, tasks: List[TaskItem]) -> Dict[str, Any]:
    by_type: Dict[str, int] = defaultdict(int)
    for task in tasks:
        by_type[task.type] += 1
    return {
        "dataset_root": dataset_root,
        "selected_task_count": len(tasks),
        "selected_task_types": sorted(by_type.keys()),
        "by_type": dict(sorted(by_type.items())),
    }


def _revision_cost_control(
    provider: str,
    model: str,
    expected_request_count: int,
    secrets: Dict[str, Any],
    estimate_cost_flag: bool,
) -> Dict[str, Any]:
    cost_control: Dict[str, Any] = {
        "expected_request_count": expected_request_count,
        "estimate_cost_enabled": estimate_cost_flag,
        "approximate_cost_usd": None,
        "pricing_source": "local_config_pricing" if (secrets.get("pricing") or {}) else None,
    }
    pricing = (secrets.get("pricing") or {}).get(provider, {})
    model_pricing = pricing.get(model) or pricing.get(model.lower())
    if isinstance(model_pricing, dict) and "per_request_usd" in model_pricing:
        cost_control["approximate_cost_usd"] = expected_request_count * float(
            model_pricing["per_request_usd"]
        )
    elif isinstance(model_pricing, dict):
        cost_control["unavailable_reason"] = (
            "pricing metadata lacks per_request_usd pre-call estimate"
        )
    else:
        cost_control["unavailable_reason"] = "pricing metadata not provided for provider/model"
    return cost_control


def _revision_dependency_versions() -> Dict[str, str]:
    return collect_dependency_versions(
        [
            "openai",
            "anthropic",
            "google-genai",
            "Pillow",
            "PyYAML",
            "tqdm",
            "numpy",
            "pandas",
            "pydantic",
        ]
    )


               
          

    
def run_eval(
    dataset_root: str,
    types: List[str],
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    max_per_type: int = 15,
    out_csv: str = "results.csv",
    secrets_file: str = "./secrets.yaml",
    stream: bool = True,
    estimate_cost_flag: bool = False,             
    timeout_sec: float = 600.0,
    seed: int = 1234,
    prompts_file: Optional[str] = None,
    prompt_prefix: str = "",
    prompt_suffix: str = "",
    prompt_mode: str = "auto",
    summary_csv: Optional[str] = None,
    thinking: bool = False,
    thinking_options: Optional[Dict[str, Any]] = None,
    collect_tokens: bool = False,
    token_log_path: Optional[str] = None,
    token_summary_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    collect_errors: bool = False,
    error_analysis_dir: Optional[str] = None,
    error_experiment_name: Optional[str] = None,
    collect_reasoning: bool = False,
    few_shot_config: Optional[Dict] = None,
    few_shot_file: str = "./few_shot_examples.yaml",
    few_shot_assets_root: str = "./few_shot_assets",
    *,
    revision_run_id: Optional[str] = None,
    revision_output_root: str = "./results/local_runs",
    write_attempts: bool = False,
    overwrite_revision_output: bool = False,
    resume_revision_output: bool = False,
) -> Dict[str, Any]:
    """
    Experiment 1: Only output aggregated results (one row per type) to out_csv;
    Console still prints per-type and overall summary. No longer writes one row per question.
    Supports few-shot learning.
    """
    if revision_run_id and not write_attempts:
        raise ValueError("revision runs require write_attempts=True")

    secrets = load_secrets(secrets_file)
    random.seed(seed)

                                                         
    prompts_cfg = _load_prompts_yaml(prompts_file) if prompts_file else {}

    if thinking_options is not None and not isinstance(thinking_options, dict):
        raise TypeError("thinking_options must be dict or None")

                   
    few_shot_examples_db = None
    exclude_examples = None
    few_shot_enabled = few_shot_config and few_shot_config.get("enabled", False)
    n_shot_limit = None

    if few_shot_enabled:
        print(f"[INFO] Few-shot learning enabled")
                          
        few_shot_examples_db = load_few_shot_examples(few_shot_file)
        try:
            ns = int(few_shot_config.get("n_shot", 0))
            if ns > 0:
                n_shot_limit = ns
        except Exception:
            n_shot_limit = None

        if few_shot_examples_db:
                               
            exclude_examples = {}
            for task_type in types:
                if task_type in few_shot_examples_db:
                    examples = few_shot_examples_db[task_type].get("examples", [])
                    if isinstance(n_shot_limit, int) and n_shot_limit > 0:
                        examples = examples[:n_shot_limit]
                    excluded_files = [ex.get("filename") for ex in examples if ex.get("filename")]
                    if excluded_files:
                        exclude_examples[task_type] = excluded_files
            print(f"[INFO] Excluding {sum(len(v) for v in exclude_examples.values())} few-shot examples from testing")

    tasks = build_tasks(dataset_root, types, max_per_type,
                        prompts_cfg=prompts_cfg,
                        prompt_prefix=prompt_prefix,
                        prompt_suffix=prompt_suffix,
                        prompt_mode=prompt_mode,
                        exclude_examples=exclude_examples)
    print(f"[INFO] Will evaluate {len(tasks)} question (types={types}）")

    revision_writer = None
    existing_revision_attempt_ids: set[str] = set()
    tasks_to_evaluate = tasks
    if revision_run_id:
        revision_writer = RevisionArtifactWriter(
            revision_output_root,
            revision_run_id,
            overwrite=overwrite_revision_output,
            resume=resume_revision_output,
        )
        existing_revision_attempt_ids = {
            attempt.attempt_id for attempt in revision_writer.iter_attempts()
        }
        if resume_revision_output and existing_revision_attempt_ids:
            tasks_to_evaluate = [
                task
                for task in tasks
                if f"{revision_run_id}:{task.type}:{task.puzzle_id}:1"
                not in existing_revision_attempt_ids
            ]
            skipped = len(tasks) - len(tasks_to_evaluate)
            if skipped:
                print(f"[INFO] Resume mode: skipping {skipped} completed attempt(s)")
        prompt_config = PromptConfig(
            prompt_mode=prompt_mode,
            prompts_file=prompts_file,
            prompts_file_sha256=sha256_file(prompts_file),
            few_shot_config=few_shot_file if few_shot_config else None,
            few_shot_config_sha256=sha256_file(few_shot_file) if few_shot_config else None,
            prompt_prefix_sha256=sha256_text(prompt_prefix),
            prompt_suffix_sha256=sha256_text(prompt_suffix),
        )
        manifest = RunManifest(
            run_id=revision_run_id,
            created_at=utc_now(),
            code_revision=collect_code_revision(),
            python_version=platform.python_version(),
            dependency_versions=_revision_dependency_versions(),
            dataset_summary=_revision_dataset_summary(dataset_root, tasks),
            prompt_config=prompt_config,
            provider=provider,
            model=model,
            seed=seed,
            retry_policy={"mode": "single_pass", "max_attempts": 1},
            cost_control=_revision_cost_control(
                provider,
                model,
                len(tasks_to_evaluate),
                secrets,
                estimate_cost_flag,
            ),
            output_paths=revision_writer.output_paths(),
        )
        revision_writer.write_manifest(manifest)

                                               
    stream_flag = (stream and (not estimate_cost_flag) and (not collect_tokens) and (not collect_errors))
    prov = None
    if tasks_to_evaluate:
        prov = make_provider(
            provider,
            model,
            secrets,
            timeout_sec,
            thinking_enabled=thinking,
            thinking_options=thinking_options
        )

    import time, csv, os
    token_tot_in = 0
    token_tot_out = 0
    token_by_type = defaultdict(lambda: {"tokens_in": 0, "tokens_out": 0, "count": 0})
    token_log_writer = None
    token_log_file = None
    if collect_tokens and token_log_path:
        os.makedirs(os.path.dirname(token_log_path) or ".", exist_ok=True)
        token_log_file = open(token_log_path, "w", encoding="utf-8", newline="")
        token_log_writer = csv.writer(token_log_file)
        token_log_writer.writerow([
            "provider", "model", "type", "puzzle_id", "tokens_in", "tokens_out", "ttft_ms", "e2e_ms"
        ])

    collector = None
    analysis_dir = None
    ea_token_log_file = None
    ea_token_log_writer = None
    if collect_errors:
        if not ERROR_COLLECTOR_AVAILABLE:
            print("⚠️ SimpleErrorCollector unavailable, skipping error analysis collection")
            collect_errors = False
        else:
            exp_name = error_experiment_name or experiment_name or "exp"
            root_dir = error_analysis_dir or "./error_analysis"
            analysis_dir = os.path.join(root_dir, exp_name)
            os.makedirs(analysis_dir, exist_ok=True)
            collector = SimpleErrorCollector(experiment_name=exp_name)
            ea_token_log_path = os.path.join(analysis_dir, "token_log.csv")
            ea_token_log_file = open(ea_token_log_path, "w", encoding="utf-8", newline="")
            ea_token_log_writer = csv.writer(ea_token_log_file)
            ea_token_log_writer.writerow([
                "provider", "model", "type", "puzzle_id", "tokens_in", "tokens_out", "ttft_ms", "e2e_ms"
            ])

              
    agg = defaultdict(lambda: {"n": 0, "ok": 0, "e2e_sum": 0.0, "ttft_sum": 0.0})
    failures: List[Dict[str, Any]] = []

    ok = 0
    errors = 0
    sum_e2e = 0.0
    wall_t0 = time.perf_counter()

                       
    for task in tqdm(tasks_to_evaluate, desc="Evaluating", ncols=0):
        schema = build_json_schema(task.type, include_reasoning=collect_reasoning)
        revision_attempt_id = (
            f"{revision_run_id}:{task.type}:{task.puzzle_id}:1" if revision_run_id else None
        )
        if revision_attempt_id and revision_attempt_id in existing_revision_attempt_ids:
            continue

                                      
        few_shot_content = None
        if few_shot_enabled and few_shot_examples_db:
            few_shot_content = build_few_shot_content(
                task.type,
                few_shot_examples_db,
                dataset_root=dataset_root,
                few_shot_assets_root=few_shot_assets_root,
                n_shot=n_shot_limit
            )

        raw, parsed, meta = prov.infer(
            prompt=task.prompt,
            images=task.images,
            json_schema=schema,
            stream=stream_flag,
            few_shot_examples=few_shot_content
        )

        failure_msg = None

        if isinstance(raw, str) and raw.startswith("__ERROR__"):
            parsed = None
            errors += 1

        passed = evaluate_pass1(task, parsed)
        ok += int(passed)

        reasoning_text = None
        if collect_reasoning and isinstance(parsed, dict):
            r = parsed.get("reasoning")
            if isinstance(r, str) and r.strip():
                reasoning_text = r.strip()

        if not passed:
            failure_msg = _describe_failure(task, parsed, raw)
            detail = {
                "type": task.type,
                "puzzle_id": task.puzzle_id,
                "explain": failure_msg,
                "raw": raw
            }
            if reasoning_text:
                detail["reasoning"] = reasoning_text
            failures.append(detail)

        e2e = float(meta.get("e2e_ms") or 0.0)
        ttft = float(meta.get("ttft_ms") or e2e)
        sum_e2e += e2e

        tokens_in = int(meta.get("tokens_in") or 0)
        tokens_out = int(meta.get("tokens_out") or 0)
        token_tot_in += tokens_in
        token_tot_out += tokens_out
        task_cost_usd = estimate_cost(provider, model, tokens_in, tokens_out, secrets)
        if revision_writer is not None and write_attempts:
            revision_writer.append_attempt(
                AttemptRecord(
                    run_id=revision_run_id,
                    attempt_id=revision_attempt_id,
                    task_type=task.type,
                    puzzle_id=task.puzzle_id,
                    attempt_index=1,
                    prompt_mode=prompt_mode,
                    provider=provider,
                    model=model,
                    parsed_answer=parsed,
                    correct=passed,
                    error_category="parse_error" if parsed is None else None,
                    latency_ms=e2e,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=task_cost_usd,
                    timestamp=utc_now(),
                )
            )
            existing_revision_attempt_ids.add(revision_attempt_id)
        rec_tok = token_by_type[task.type]
        rec_tok["tokens_in"] += tokens_in
        rec_tok["tokens_out"] += tokens_out
        rec_tok["count"] += 1
        if token_log_writer:
            token_log_writer.writerow([
                provider,
                model,
                task.type,
                task.puzzle_id,
                tokens_in,
                tokens_out,
                f"{ttft:.1f}",
                f"{e2e:.1f}"
            ])

        if collector:
                                                   
            error_description = failure_msg if not passed else None

            reasoning_field = None
            if collect_reasoning and isinstance(parsed, dict):
                r = parsed.get("reasoning")
                if isinstance(r, str) and r.strip():
                    reasoning_field = r.strip()

            collector.record(
                task_type=task.type,
                puzzle_id=task.puzzle_id,
                prompt=task.prompt,
                gt=task.gt,
                raw=raw,
                parsed=parsed,
                pass1=passed,
                meta=meta,
                error_description=error_description,
                reasoning=reasoning_field
            )
            if ea_token_log_writer:
                ea_token_log_writer.writerow([
                    provider,
                    model,
                    task.type,
                    task.puzzle_id,
                    tokens_in,
                    tokens_out,
                    f"{ttft:.1f}",
                    f"{e2e:.1f}"
                ])

        rec = agg[task.type]
        rec["n"] += 1
        rec["ok"] += int(passed)
        rec["e2e_sum"] += e2e
        rec["ttft_sum"] += ttft

    if token_log_file:
        token_log_file.flush()
        token_log_file.close()
    if ea_token_log_file:
        ea_token_log_file.flush()
        ea_token_log_file.close()

    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
    pass1 = ok/len(tasks) if tasks else 0.0

                        
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    overall_cost = estimate_cost(provider, model, token_tot_in, token_tot_out, secrets) or 0.0

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
                                 
        writer.writerow([
            "provider","model","type","n","pass_at_1","avg_ttft_ms","avg_e2e_ms",
            "tokens_in","tokens_out","total_tokens","cost_usd","cost_per_question"
        ])
        for t in sorted(agg.keys()):
            s = agg[t]
            n = s["n"]
            pass_at_1 = (s["ok"]/n) if n else 0.0
            avg_ttft = (s["ttft_sum"]/n) if n else 0.0
            avg_e2e  = (s["e2e_sum"]/n) if n else 0.0
            tok = token_by_type[t]
            tokens_in = tok["tokens_in"]
            tokens_out = tok["tokens_out"]
            total_tokens = tokens_in + tokens_out
            cost_usd = estimate_cost(provider, model, tokens_in, tokens_out, secrets) or 0.0
            cost_per_question = (cost_usd / n) if n else 0.0
            writer.writerow([
                provider,
                model,
                t,
                n,
                f"{pass_at_1:.3f}",
                f"{avg_ttft:.1f}",
                f"{avg_e2e:.1f}",
                tokens_in,
                tokens_out,
                total_tokens,
                f"{cost_usd:.6f}",
                f"{cost_per_question:.6f}"
            ])

                    
    print("\n[SUMMARY by type]")
    for t in sorted(agg.keys()):
        s = agg[t]; n = s["n"]; ok_t = s["ok"]
        e2e_avg = (s["e2e_sum"]/n) if n else 0.0
        ttft_avg = (s["ttft_sum"]/n) if n else 0.0
        tok = token_by_type[t]
        tokens_in = tok["tokens_in"]
        tokens_out = tok["tokens_out"]
        cost_usd = estimate_cost(provider, model, tokens_in, tokens_out, secrets) or 0.0
        print(f"- {t:<18} n={n:3d}  pass@1={ok_t/n if n else 0:.3f}  "
              f"avg_e2e_ms={e2e_avg:.1f}  avg_ttft_ms={ttft_avg:.1f}  "
              f"tokens_in={tokens_in} tokens_out={tokens_out} cost_usd={cost_usd:.6f}")

    summary_path = None
    if collect_tokens and token_summary_path:
        summary_path = token_summary_path
    elif collect_errors and analysis_dir:
        summary_path = os.path.join(analysis_dir, "token_summary.json")

    if summary_path:
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        summary = {
            "experiment": experiment_name or "",
            "provider": provider,
            "model": model,
            "overall": {
                "total_question": len(tasks),
                "total_tokens_in": token_tot_in,
                "total_tokens_out": token_tot_out,
                "total_tokens": token_tot_in + token_tot_out,
                "cost_usd": overall_cost
            },
            "by_task_type": {
                t: {
                    "count": tok["count"],
                    "tokens_in": tok["tokens_in"],
                    "tokens_out": tok["tokens_out"],
                    "total_tokens": tok["tokens_in"] + tok["tokens_out"],
                    "cost_usd": (estimate_cost(provider, model, tok["tokens_in"], tok["tokens_out"], secrets) or 0.0)
                }
                for t, tok in sorted(token_by_type.items())
            }
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Token summary written to {summary_path}")

    print(f"\n[OVERALL] tasks={len(tasks)}  pass@1={pass1:.3f}  sum_e2e_ms={sum_e2e:.1f}  wall_ms={wall_ms:.1f}")
    print(f"[OVERALL TOKENS] tokens_in={token_tot_in} tokens_out={token_tot_out} cost_usd={overall_cost:.6f}")

                                 
    if summary_csv:
        with open(summary_csv, "w", encoding="utf-8") as f:
            f.write("type,n,pass,pass_rate,e2e_avg_ms,ttft_avg_ms\n")
            for t in sorted(agg.keys()):
                s = agg[t]; n = s["n"]; ok_t = s["ok"]
                e2e_avg = (s["e2e_sum"]/n) if n else 0.0
                ttft_avg = (s["ttft_sum"]/n) if n else 0.0
                f.write(
                    f"{t},{n},{ok_t},{(ok_t/n if n else 0):.6f},"
                    f"{e2e_avg:.3f},{ttft_avg:.3f}\n"
                )

    if revision_writer is not None:
        revision_writer.write_summaries_from_attempts()

    print(f"[DONE] Pass@1 = {ok}/{len(tasks)} = {pass1:.3f} ; errors={errors}. Results saved to {out_csv}")
    result = {
        "n": len(tasks), "pass1": pass1, "errors": errors,
        "out_csv": out_csv, "provider": provider, "model": model,
        "sum_e2e_ms": sum_e2e, "wall_ms": wall_ms,
        "tokens_in": token_tot_in,
        "tokens_out": token_tot_out,
        "cost_usd": overall_cost,
        "by_type": {
            t: {
                "n": s["n"],
                "ok": s["ok"],
                "pass_at_1": (s["ok"]/s["n"] if s["n"] else 0.0),
                "e2e_avg_ms": (s["e2e_sum"]/s["n"] if s["n"] else 0.0),
                "ttft_avg_ms": (s["ttft_sum"]/s["n"] if s["n"] else 0.0),
                "tokens_in": token_by_type[t]["tokens_in"],
                "tokens_out": token_by_type[t]["tokens_out"],
                "cost_usd": estimate_cost(provider, model, token_by_type[t]["tokens_in"], token_by_type[t]["tokens_out"], secrets) or 0.0,
            } for t, s in agg.items()
        }
    }
    result["failures"] = failures
    if collector and analysis_dir:
        collector.save_summary(analysis_dir)
        result["error_analysis_dir"] = analysis_dir
    return result


    
           

def run_until_type_correct(
    dataset_root: str,
    provider: str,
    model: str,
    types: List[str],
    max_attempt_per_type: int = 6,                           
    max_pool_per_type: int = 50,                                
    use_full_dataset_pool: bool = True,                              
    secrets_file: str = "./secrets.yaml",
    timeout_sec: float = 120.0,
    prompts_file: Optional[str] = None,
    prompt_mode: str = "auto",
    prompt_prefix: str = "",
    prompt_suffix: str = "",
    out_csv: str = "until_type_correct.csv",
    log_attempt_rows: bool = False,                           
    retry_sleep_ms: int = 150,
    cache_bust: bool = True,
    stream: bool = False,
    thinking: bool = False,
    thinking_options: Optional[Dict[str, Any]] = None,
    collect_tokens: bool = False,
    token_log_path: Optional[str] = None,
    token_summary_path: Optional[str] = None,
    collect_reasoning: bool = False
) -> Dict[str, Any]:
    """
    Newexperiment：by“task type”asunit，with replacementsampleattemptquestions；ifonequestionfailthencontinuesamplenext one，
    until first success for that task type or reaching max attempt.
    Progress indicators:
      - eachenteronetask typewhenprint“Startevaluatecertaintask type（maxNattempts）”
      - eachattemptsattemptbeforeprint“[Type] Attempt a/b • PID=...”
      - eachattemptsattemptafterprint“↳ OK/FAIL/ERROR  e2e=..ms  cum=..ms”
      - eachtask typeEndprint“Summary：attempt / cum_ms / success andfirsthitPID”
    Args:
        thinking: Whether to enable thinking/reasoning mode for each Provider
        thinking_options: Additional configuration for thinking mode
        collect_tokens: Whether to collect tokens and write to log/summary
        token_log_path: Output detailed token log CSV if provided
        token_summary_path: Output token summary JSON if provided
        collect_reasoning: Whether to require model to return reasoning field
    """
    import os, uuid, time, random, csv

                   
    secrets = load_secrets(secrets_file)
    prompts_cfg = _load_prompts_yaml(prompts_file) if prompts_file else {}
    prov = make_provider(
        provider,
        model,
        secrets,
        timeout_sec,
        thinking_enabled=thinking,
        thinking_options=thinking_options or {}
    )

    token_tot_in = 0
    token_tot_out = 0
    token_by_type = defaultdict(lambda: {"tokens_in": 0, "tokens_out": 0})
    token_log_writer = None
    token_log_file = None
    if collect_tokens and token_log_path:
        os.makedirs(os.path.dirname(token_log_path) or ".", exist_ok=True)
        token_log_file = open(token_log_path, "w", encoding="utf-8", newline="")
        token_log_writer = csv.writer(token_log_file)
        token_log_writer.writerow([
            "phase", "provider", "model", "type", "puzzle_id", "attempt_idx",
            "tokens_in", "tokens_out", "ttft_ms", "e2e_ms", "reasoning"
        ])

                             
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    csv_rows = []             
    csv_rows.append("kind,provider,model,type,puzzle_id,attempt_idx,cumulative_ms,pass1,notes")              

    overall = {"by_type": {}, "sum_attempt": 0, "sum_cum_ms": 0.0, "sum_pass1": 0, "n_types": 0}

    for t in types:
        print(f"\n========== [{t}] Starting evaluation (max {max_attempt_per_type} attempt) ==========", flush=True)

                   
                                                          
                                         
        effective_max = None if use_full_dataset_pool else max_pool_per_type
        pool_tasks = build_tasks(
            dataset_root=dataset_root,
            types=[t],
            max_per_type=effective_max,
            prompts_cfg=prompts_cfg,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
            prompt_mode=prompt_mode
        )
        random.shuffle(pool_tasks)

        if not pool_tasks:
            print(f"[{t}] ⚠️ No available question, skipping this type")
            continue

        attempt = 0
        cumulative = 0.0
        success = 0
        first_success_pid = ""
        last_err = ""

        while attempt < max_attempt_per_type:
            task = random.choice(pool_tasks)         

            attempt += 1
            schema = build_json_schema(task.type)

            pr = task.prompt
            if cache_bust:
                pr = pr + f"\n\n[IGNORE THIS LINE] nonce={uuid.uuid4().hex} attempt={attempt}"

            print(f"[{t}] Attempt {attempt}/{max_attempt_per_type} • PID={task.puzzle_id}", flush=True)

            try:
                raw, parsed, meta = prov.infer(
                    prompt=pr, images=task.images, json_schema=schema, stream=stream
                )
                e2e = float(meta.get("e2e_ms") or 0.0)
                cumulative += e2e
                ok = evaluate_pass1(task, parsed)
                status = "OK" if ok else "FAIL"
                print(f" ↳ {status}  e2e={e2e:.1f}ms  cum={cumulative:.1f}ms", flush=True)
            except Exception as e:
                ok = False
                last_err = f"{type(e).__name__}: {e}"
                meta = {"e2e_ms": 0.0, "ttft_ms": 0.0, "tokens_in": 0, "tokens_out": 0}
                print(f" ↳ ERROR {last_err}", flush=True)

            tokens_in = int((meta.get("tokens_in") or 0))
            tokens_out = int((meta.get("tokens_out") or 0))
            token_tot_in += tokens_in
            token_tot_out += tokens_out
            token_by_type[t]["tokens_in"] += tokens_in
            token_by_type[t]["tokens_out"] += tokens_out

            reasoning_txt = None
            if collect_reasoning and isinstance(parsed, dict):
                r = parsed.get("reasoning")
                if isinstance(r, str) and r.strip():
                    reasoning_txt = r.strip()

            if token_log_writer:
                token_log_writer.writerow([
                    "attempt",
                    provider,
                    model,
                    t,
                    task.puzzle_id,
                    attempt,
                    tokens_in,
                    tokens_out,
                    f"{meta.get('ttft_ms', 0.0):.1f}",
                    f"{e2e:.1f}",
                    reasoning_txt or ""
                ])

            if log_attempt_rows:
                csv_rows.append(f"attempt,{provider},{model},{t},{task.puzzle_id},{attempt},{cumulative:.1f},{int(ok)},{last_err}")

            if ok:
                success = 1
                first_success_pid = task.puzzle_id
                print(f"[{t}] 🎯 First hit: PID={first_success_pid}  Cumulative time={cumulative:.1f}ms  Attempts={attempt}", flush=True)
                break

            time.sleep(retry_sleep_ms / 1000.0)

               
        csv_rows.append(f"summary,{provider},{model},{t},{first_success_pid},{attempt},{cumulative:.1f},{success},{last_err}")

        if token_log_writer:
            token_log_writer.writerow([
                "summary",
                provider,
                model,
                t,
                first_success_pid,
                attempt,
                token_by_type[t]["tokens_in"],
                token_by_type[t]["tokens_out"],
                "",
                f"{cumulative:.1f}",
                ""
            ])

        print(f"[{t}] Summary: attempt={attempt}  cum_ms={cumulative:.1f}  success={success}  first_pid={first_success_pid or '-'}", flush=True)

        overall["by_type"][t] = {
            "attempt": attempt,
            "cumulative_ms": cumulative,
            "pass1": success,
            "first_success_pid": first_success_pid
        }
        overall["sum_attempt"] += attempt
        overall["sum_cum_ms"] += cumulative
        overall["sum_pass1"] += success
        overall["n_types"] += 1

                          
    with open(out_csv, "w", encoding="utf-8") as outp:
        for row in csv_rows:
            outp.write(row + "\n")

                          
    n = max(1, overall["n_types"])
    overall["avg_attempt_per_type"] = overall["sum_attempt"] / n
    overall["avg_cum_ms_per_type"] = overall["sum_cum_ms"] / n
    overall["pass_rate_types"] = overall["sum_pass1"] / n
    overall["out_csv"] = out_csv
    overall["tokens_in"] = token_tot_in
    overall["tokens_out"] = token_tot_out

    print(f"\n[UNTIL-TYPE] types={types}  pass_rate={overall['pass_rate_types']:.3f}  "
          f"avg_attempt={overall['avg_attempt_per_type']:.2f}  "
          f"avg_cum_ms={overall['avg_cum_ms_per_type']:.1f}  -> {out_csv}", flush=True)

    if token_log_file:
        token_log_file.flush()
        token_log_file.close()

    if collect_tokens and token_summary_path:
        os.makedirs(os.path.dirname(token_summary_path) or ".", exist_ok=True)
        with open(token_summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "provider": provider,
                "model": model,
                "overall": {
                    "tokens_in": token_tot_in,
                    "tokens_out": token_tot_out,
                    "total_tokens": token_tot_in + token_tot_out
                },
                "by_type": {
                    typ: {
                        "tokens_in": data["tokens_in"],
                        "tokens_out": data["tokens_out"],
                        "total_tokens": data["tokens_in"] + data["tokens_out"]
                    } for typ, data in token_by_type.items()
                }
            }, f, indent=2, ensure_ascii=False)

    return overall

    
def main():
    """
    CLI entry: parse arguments and call run_eval (supports prompts-file / prefix / suffix)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="./captcha_data")
    parser.add_argument("--types", nargs="+", required=True, help="Task types to evaluate (e.g., Dice_Count Geometry_Click ...)")
    parser.add_argument("--provider", default="openai", help="openai|anthropic|gemini|fireworks|openrouter")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument("--max-per-type", type=int, default=15, help="Max question per type")
    parser.add_argument("--out-csv", default="results.csv", help="Output CSV path")
    parser.add_argument("--secrets-file", default="./secrets.yaml", help="Secrets config file path")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming (for usage/cost estimation)")
    parser.add_argument("--estimate-cost", action="store_true", help="Enable cost estimation (usually requires non-streaming)")
    parser.add_argument("--timeout-sec", type=float, default=120.0, help="Timeout seconds per request")
    
    parser.add_argument("--prompts-file", default=None, help="Prompt override file (yaml/json), key is task type or default")
    parser.add_argument("--prompt-prefix", default="", help="String to prepend to each question prompt")
    parser.add_argument("--prompt-suffix", default="", help="String to append to each question prompt")
    parser.add_argument("--prompt-mode", choices=["auto","gt","opt"], default="auto",
                        help="Prompt source: gt=use GT original; opt=use prompts.yaml; auto=GT>yaml>default")
    parser.add_argument("--summary-csv", default=None, help="Save per-type summary CSV separately")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking/reasoning extension for each Provider (if supported).")
    parser.add_argument("--thinking-config", default=None, help="Thinking configuration (JSON string), e.g., '{\"effort\":\"medium\"}'.")
    parser.add_argument("--collect-reasoning", action="store_true",
                        help="Require model to output reasoning field (may increase time and cost)")
    parser.add_argument("--revision-run-id", default=None, help="Write revision artifacts under this run id")
    parser.add_argument("--revision-output-root", default="./results/local_runs", help="Revision artifact output root")
    parser.add_argument("--write-attempts", action="store_true", help="Write append-only revision attempt records")
    parser.add_argument("--overwrite-revision-output", action="store_true", help="Replace an existing revision run directory")
    parser.add_argument("--resume-revision-output", action="store_true", help="Reuse an existing revision run directory")
    
    parser.add_argument("--until-correct-type", default=None, 
                        help="Only execute 'until-correct' experiment for this type, e.g., Dice_Count. When this is None, run regular evaluation; otherwise use run_until_correct(...)")
    parser.add_argument("--max-attempt", type=int, default=10, help="'until-correct' max attempt count")
    parser.add_argument("--retry-sleep-ms", type=int, default=200, help="'until-correct' interval between attempt in milliseconds")
    parser.add_argument("--no-cache-bust", action="store_true", help="Disable 'until-correct' cache-bust nonce")


    args = parser.parse_args()

    thinking_options = None
    if args.thinking_config:
        try:
            thinking_options = json.loads(args.thinking_config)
            if not isinstance(thinking_options, dict):
                raise ValueError("thinking_config must be a JSON object")
        except Exception as exc:
            raise SystemExit(f"Failed to parse --thinking-config: {exc}")
    thinking_enabled = args.enable_thinking or bool(thinking_options)

    if args.until_correct_type:
                                                          
        run_until_type_correct(
            dataset_root=args.dataset_root,
            provider=args.provider,
            model=args.model,
            types=[args.until_correct_type],
            max_attempt_per_type=args.max_attempt,
            max_pool_per_type=args.max_per_type,
            secrets_file=args.secrets_file,
            timeout_sec=args.timeout_sec,
            prompts_file=args.prompts_file,
            prompt_mode=args.prompt_mode,
            prompt_prefix="",                            
            prompt_suffix="",
            out_csv=args.out_csv,
            retry_sleep_ms=args.retry_sleep_ms,
            cache_bust=(not args.no_cache_bust),
            stream=(not args.no_stream),
            thinking=thinking_enabled,
            thinking_options=thinking_options
        )
    else:
                                                                    
        run_eval(
            dataset_root=args.dataset_root,
            types=args.types,
            provider=args.provider,
            model=args.model,
            max_per_type=args.max_per_type,
            out_csv=args.out_csv,
            secrets_file=args.secrets_file,
            stream=(not args.no_stream),
            estimate_cost_flag=args.estimate_cost,
            timeout_sec=args.timeout_sec,
            prompts_file=args.prompts_file,
            prompt_prefix="",
            prompt_suffix="",
            prompt_mode=args.prompt_mode,                         
            summary_csv=args.summary_csv,
            thinking=thinking_enabled,
            thinking_options=thinking_options,
            collect_reasoning=args.collect_reasoning,
            revision_run_id=args.revision_run_id,
            revision_output_root=args.revision_output_root,
            write_attempts=args.write_attempts,
            overwrite_revision_output=args.overwrite_revision_output,
            resume_revision_output=args.resume_revision_output,
        )


if __name__ == "__main__":
    raise SystemExit(main())
