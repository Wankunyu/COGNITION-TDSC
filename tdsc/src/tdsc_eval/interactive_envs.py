"""Action-dependent A/D interfaces around the unchanged source task scorers.

The puzzle retains its pixels and coordinate origin. Controls occupy an appended
footer; references remain separate observations in their original image order.
No label is consulted until submission, and observations only depict actions.
"""
import copy
import importlib
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .calibration import native_modules
from .tasks import ROOT, score


INTEGER_TYPES = {"Dice_Count", "Visible_Eye_Count"}
POINT_TYPES = {"Place_Dot", "Pick_Area"}
MULTIPOINT_TYPES = {"Click_Order", "Select_Animal_Optimized"}
GRID_TYPES = {"Patch_Select", "Select_Animal"}
SUPPORTED_TYPES = INTEGER_TYPES | POINT_TYPES | MULTIPOINT_TYPES | GRID_TYPES | {"Rotation_Match"}


class InvalidActionError(ValueError):
    """A parsed model action has invalid parameters for the adapted interface."""


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _inside(point, box):
    return box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]


def _font(size):
    for filename in ("/System/Library/Fonts/SFNSMono.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(filename, size=size)
        except OSError:
            continue
    raise RuntimeError("A scalable UI font is required for the adapted environment")


def _task_instruction(row):
    kind = row["type"]
    if row.get("module") == "D":
        # Defense manifests carry task semantics only; output instructions are
        # supplied by each protocol. The target is visible, never inferred from GT.
        return row["prompt"].strip()
    instructions = {
        "Dice_Count": (
            "Sum the values of all dice TOP faces (the horizontal faces pointing upward). "
            "Use a visible digit as its value and count visible pips as dots. For partial "
            "occlusion count only clearly visible digits/pips on the TOP face; do not guess "
            "hidden parts. Ignore decorations, logos, shadows and background patterns."
        ),
        "Click_Order": (
            "Image 1 is the REFERENCE sequence; image 2 is the MAIN puzzle. "
            "Click near the center of every matching icon in the MAIN image, in exactly "
            "the order shown in the REFERENCE. Coordinates use the MAIN image origin."
        ),
        "Place_Dot": (
            "Place a dot at the endpoint of the single continuous path connected to the "
            "large start marker with the car icon. Follow the path regardless of color; "
            "ignore unconnected shapes. The start marker is not the target. For a line "
            "tip use its center; for an ending area use its geometric center; at an "
            "image edge use the last visible in-bounds pixel."
        ),
        "Pick_Area": (
            "Click strictly inside the LARGEST explicitly outlined area, near its "
            "geometric center. If two areas appear almost equal, choose the one with "
            "the clearly larger enclosed region."
        ),
        "Rotation_Match": (
            "Image 1 is the REFERENCE direction; image 2 is the CURRENT object. "
            "Determine the additional CLOCKWISE rotation in degrees, in [0, 360), "
            "needed to make the object's facing direction match the reference."
        ),
    }
    if kind == "Patch_Select":
        # The target/category comes from the public prompt, not correct_patches.
        target = row["prompt"].split("Indexing Rules:", 1)[0].strip()
        return (target + "\nSelect ALL cells matching the target pattern or reference. "
                "Use category, shape and key parts as primary cues; use color and "
                "texture as supporting evidence. Leave all cells unselected if none match.")
    if row.get("source") == "interactive_dev_v1":
        return row["prompt"].strip()
    return instructions[kind]


class AdaptedEnv:
    """Small local interface implementing exactly the A/D task action contract."""

    def __init__(self, row, episode_dir):
        if row["type"] not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported adapted task: {row['type']}")
        self._row = copy.deepcopy(row)
        self.kind = row["type"]
        self.output = Path(episode_dir) / "adapted_observations"
        self.output.mkdir(parents=True, exist_ok=True)
        self.layout = copy.deepcopy(row.get("layout", {}))
        self.main_index = self.layout.get(
            "main_image_index", 1 if self.kind in {"Click_Order", "Rotation_Match"} else 0)
        self.source_paths = [Path(path) if Path(path).is_absolute() else ROOT / path
                             for path in row["images"]]
        self.originals = []
        for path in self.source_paths:
            with Image.open(path) as image:
                self.originals.append(image.convert("RGB"))
        if not 0 <= self.main_index < len(self.originals):
            raise ValueError("main_image_index is outside the observation list")
        self.width, self.height = self.originals[self.main_index].size
        expected_sizes = row.get("image_sizes")
        if expected_sizes and [list(image.size) for image in self.originals] != expected_sizes:
            raise ValueError("Manifest image dimensions do not match source images")
        self.footer_height = max(120, self.width // 8)
        self.canvas_width = max(self.width, 300)
        margin = max(12, self.canvas_width // 50)
        button_width = min(max(100, self.canvas_width // 6), self.canvas_width // 3)
        top = self.height + self.footer_height // 2
        bottom = self.height + self.footer_height - margin
        self.controls = {"submit": [self.canvas_width - margin - button_width,
                                    top, self.canvas_width - margin, bottom]}
        self.input_box = None
        if self.kind in INTEGER_TYPES | {"Rotation_Match"}:
            self.input_box = [margin, top, self.controls["submit"][0] - margin, bottom]
        if self.kind == "Select_Animal_Optimized":
            self.controls["undo"] = [margin, top, margin + button_width, bottom]
        self.grid = None
        if self.kind in GRID_TYPES:
            if all(key in self.layout for key in ("rows", "cols", "grid_box")):
                rows, cols = self.layout["rows"], self.layout["cols"]
                box = self.layout["grid_box"]
                origin = "manifest.layout"
            else:
                # Historical A prompts explicitly describe the visible 5 x 5
                # grid. Do not recover layout from the answer-side grid_size.
                grids = set(re.findall(r"\b(\d+)\s*[×xX]\s*(\d+)\s+grid\b", row["prompt"]))
                if len(grids) != 1:
                    raise ValueError("Grid task needs independent visible layout metadata")
                rows, cols = map(int, next(iter(grids)))
                box = [0, 0, self.width, self.height]
                origin = "public_prompt_grid_dimensions"
            if (type(rows) is not int or type(cols) is not int or rows <= 0 or cols <= 0
                    or not isinstance(box, (list, tuple)) or len(box) != 4
                    or not all(_finite(value) for value in box)
                    or not (0 <= box[0] < box[2] <= self.width
                            and 0 <= box[1] < box[3] <= self.height)):
                raise ValueError("Invalid public grid layout")
            self.grid = {"rows": rows, "cols": cols, "grid_box": list(box), "source": origin}
        native_modules()
        self._response_type = importlib.import_module("captcha.data_types").EnvResponse
        self._reset_state()

    def _reset_state(self):
        self.points = []
        self.indices = set()
        self.number = None
        self.done = False
        self.reward = 0.0
        self.step_index = 0

    def public_metadata(self):
        result = {"environment": "cognition_adapted_v1.1_numeric_input_box", "main_image_index": self.main_index,
                  "source_images": [str(path) for path in self.source_paths],
                  "source_image_sizes": [list(image.size) for image in self.originals],
                  "main_image_box": [0, 0, self.width, self.height],
                  "observation_main_size": [self.canvas_width, self.height + self.footer_height],
                  "coordinate_mapping": {"scale": [1, 1], "offset": [0, 0]},
                  "controls": copy.deepcopy(self.controls), "input_box": copy.deepcopy(self.input_box),
                  "grid": copy.deepcopy(self.grid)}
        return result

    def instruction(self):
        submit = self.controls["submit"]
        center = [(submit[0] + submit[2]) / 2, (submit[1] + submit[3]) / 2]
        text = (_task_instruction(self._row) + f"\n\nThe MAIN puzzle occupies x in [0, {self.width}) "
                f"and y in [0, {self.height}); its pixels and origin are unchanged. "
                f"Controls are below the puzzle in image {self.main_index + 1}. "
                "Reference images retain their original order and dimensions. ")
        if self.kind in INTEGER_TYPES:
            text += "Use enter_number with an integer to fill or replace the answer. "
        elif self.kind == "Rotation_Match":
            text += ("Use enter_number with a finite numeric angle; decimals are accepted and retained. "
                     "The entered angle is displayed; the object image does not rotate. ")
        elif self.kind in POINT_TYPES:
            text += "Click the puzzle to set your point; another puzzle click replaces it. "
        elif self.kind in GRID_TYPES:
            text += (f"The grid has {self.grid['rows']} rows and {self.grid['cols']} columns. "
                     "Click a cell to select it; click it again to cancel the selection. ")
        else:
            text += "Each puzzle click appends one point. "
            if self.kind == "Click_Order":
                text += "Numbered markers show your click order. "
            else:
                box = self.controls["undo"]
                text += (f"Click Undo at ({(box[0]+box[2])/2:g}, {(box[1]+box[3])/2:g}) "
                         "to remove the last point. Hollow markers show selected points. ")
        return (text + f"When finished, click Submit at ({center[0]:g}, {center[1]:g}). "
                "Submission ends the puzzle. Markers and filled values record your actions; "
                "they do not indicate whether an answer is correct.")

    def reset(self, task_index=0):
        if task_index != 0:
            raise ValueError("An adapted environment contains exactly one staged task")
        self._reset_state()
        return self._response()

    def final_answer(self):
        if self.kind in INTEGER_TYPES:
            return {"value": self.number} if self._row.get("module") == "D" else {
                "answer_type": "number", "value": self.number}
        if self.kind == "Rotation_Match":
            return {"answer_type": "rotation", "angle": self.number}
        if self.kind in GRID_TYPES:
            value = {"indices": sorted(self.indices)}
            if self._row.get("module") != "D":
                value["answer_type"] = "multi_select"
            return value
        if self.kind in POINT_TYPES:
            return {"answer_type": "single_point", "point": copy.deepcopy(self.points[-1]) if self.points else None}
        value = {"points": copy.deepcopy(self.points)}
        if self._row.get("module") != "D":
            value["answer_type"] = "click_order"
        return value

    def step(self, action):
        if self.done:
            raise InvalidActionError("Cannot act after submission")
        self.step_index += 1
        if action.name == "enter_number" and self.kind in INTEGER_TYPES | {"Rotation_Match"}:
            number = action.kwargs.get("number")
            if not _finite(number) or (self.kind in INTEGER_TYPES and int(number) != number):
                raise InvalidActionError("enter_number needs a finite number, integral for counting tasks")
            self.number = int(number) if self.kind in INTEGER_TYPES else number
        elif action.name == "click":
            position = action.kwargs.get("position")
            if (not isinstance(position, (list, tuple)) or len(position) != 2
                    or not all(_finite(value) for value in position)):
                raise InvalidActionError("click needs two finite pixel coordinates")
            if _inside(position, self.controls["submit"]):
                answer = self.final_answer()
                self.done = True
                if self._row.get("module") == "D":
                    from .defense_tasks import score_defense
                    self.reward = float(score_defense(self._row, answer))
                else:
                    self.reward = float(score(self._row, answer))
            elif "undo" in self.controls and _inside(position, self.controls["undo"]):
                if self.points:
                    self.points.pop()
            elif _inside(position, [0, 0, self.width, self.height]):
                point = {"x": position[0], "y": position[1]}
                if self.kind in POINT_TYPES:
                    self.points = [point]
                elif self.kind in MULTIPOINT_TYPES:
                    self.points.append(point)
                elif self.kind in GRID_TYPES and _inside(position, self.grid["grid_box"]):
                    x1, y1, x2, y2 = self.grid["grid_box"]
                    col = int((position[0] - x1) * self.grid["cols"] / (x2 - x1))
                    row = int((position[1] - y1) * self.grid["rows"] / (y2 - y1))
                    index = row * self.grid["cols"] + col
                    self.indices.symmetric_difference_update({index})
        # Other structurally parsed actions have no effect and consume a step.
        return self._response()

    def _response(self):
        return self._response_type(observation=self._render(), reward=self.reward, done=self.done)

    def _render(self):
        canvas = Image.new("RGB", (self.canvas_width, self.height + self.footer_height), "#eef1f5")
        puzzle = self.originals[self.main_index].copy()
        draw = ImageDraw.Draw(puzzle)
        # Marker thickness is in original-image pixels. D eyes retain an empty
        # center; drawing is restricted to the puzzle so labels never cover UI.
        line_width = max(2, self.width // 600)
        if self.kind in GRID_TYPES:
            x1, y1, x2, y2 = self.grid["grid_box"]
            for index in sorted(self.indices):
                r, c = divmod(index, self.grid["cols"])
                box = [x1 + c * (x2-x1) / self.grid["cols"],
                       y1 + r * (y2-y1) / self.grid["rows"],
                       x1 + (c+1) * (x2-x1) / self.grid["cols"] - 1,
                       y1 + (r+1) * (y2-y1) / self.grid["rows"] - 1]
                draw.rectangle(box, outline="#ff00bb", width=line_width * 2)
                pad = max(5, (box[2]-box[0]) // 12)
                draw.line([(box[0]+pad, box[1]+2*pad), (box[0]+2*pad, box[1]+3*pad),
                           (box[0]+4*pad, box[1]+pad)], fill="#ff00bb", width=line_width * 2)
        for index, point in enumerate(self.points):
            x, y = point["x"], point["y"]
            radius = 9 if self.kind == "Select_Animal_Optimized" else max(7, self.width // 80)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline="#ff00bb", width=line_width)
            if self.kind == "Click_Order":
                label = str(index + 1)
                draw.text((min(self.width-20, x+radius+2), max(0, y-radius)), label,
                          font=_font(max(14, self.width // 45)), fill="#ff00bb", stroke_width=1,
                          stroke_fill="white")
        canvas.paste(puzzle, (0, 0))
        draw = ImageDraw.Draw(canvas)
        font_size = max(16, self.canvas_width // 60)
        margin = max(12, self.canvas_width // 50)
        if self.kind in INTEGER_TYPES | {"Rotation_Match"}:
            status = "Clockwise angle (deg)" if self.kind == "Rotation_Match" else "Answer"
            draw.rectangle(self.input_box, fill="white", outline="#566578", width=line_width)
            if self.number is not None:
                box = self.input_box
                value = str(self.number)
                value_size = font_size
                padding = max(6, line_width * 2)
                while value_size > 1:
                    value_font = _font(value_size)
                    bounds = draw.textbbox((0, 0), value, font=value_font)
                    if (bounds[2]-bounds[0] <= box[2]-box[0]-2*padding
                            and bounds[3]-bounds[1] <= box[3]-box[1]-2*padding):
                        break
                    value_size -= 1
                draw.text(((box[0]+box[2])/2, (box[1]+box[3])/2), value,
                          anchor="mm", fill="#17212f", font=_font(value_size))
        elif self.kind in GRID_TYPES:
            status = "Selected cells: " + str(len(self.indices))
        else:
            status = "Selected points: " + str(len(self.points))
        draw.text((margin, self.height + margin), status, fill="#17212f", font=_font(font_size))
        for name, box in self.controls.items():
            draw.rounded_rectangle(box, radius=max(5, font_size//3), fill="#1d4d78", outline="#112a43")
            draw.text(((box[0]+box[2])/2, (box[1]+box[3])/2), name.title(),
                      anchor="mm", fill="white", font=_font(font_size))
        observation = []
        for index, original in enumerate(self.originals):
            path = self.output / f"observation_{self.step_index:03d}_{index}.png"
            (canvas if index == self.main_index else original).save(path)
            observation.append(str(path))
        return observation


def make_adapted_env(row, episode_dir):
    env = AdaptedEnv(row, episode_dir)
    return env, env.instruction()
