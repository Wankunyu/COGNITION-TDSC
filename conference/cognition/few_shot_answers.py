#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hard-coded answers for few-shot examples. This avoids filename drift (for
example PNG compressed to JPG) breaking lookups, so we do not rely on
ground_truth.json for these entries.

Structure:
{
    "TaskType": [
        {"filename": "...", "answer": {...}, "order_image": "...", "reference_image": "..."},
        ...
    ]
}

Keep filenames in sync with few_shot_examples.yaml and any asset renames.
"""

FEW_SHOT_ANSWERS = {
    "Bingo": [
        {
            "filename": "bingo1.jpg",
            "answer": {
                "points": [
                    {"x": 0, "y": 7},
                    {"x": 3, "y": 8}
                ]
            }
        },
        {
            "filename": "bingo2.jpg",
            "answer": {
                "points": [
                    {"x": 4, "y": 7}
                ]
            }
        }
    ],

    "Click_Order": [
        {
            "filename": "image1.jpg",
            "order_image": "order1.jpg",
            "tolerance": 40,
            "answer": {
                "points": [
                    {"x": 470, "y": 100},
                    {"x": 325, "y": 180},
                    {"x": 162, "y": 100},
                    {"x": 190, "y": 275}
                ]
            }
        },
        {
            "filename": "image2.jpg",
            "order_image": "order2.jpg",
            "tolerance": 40,
            "answer": {
                "points": [
                    {"x": 520, "y": 170},
                    {"x": 300, "y": 80},
                    {"x": 125, "y": 310},
                    {"x": 500, "y": 310}
                ]
            }
        }
    ],

    "Connect_icon": [
        {
            "filename": "puzzle1.json",
            "reference_image": "reference1.jpg",
            "options": ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg", "image6.jpg", "image7.jpg"],
            "answer": {
                "choice": 0
            }
        },
        {
            "filename": "puzzle2.json",
            "reference_image": "reference2.jpg",
            "options": ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg", "image6.jpg", "image7.jpg"],
            "answer": {
                "choice": 4
            }
        }
    ],

    "Coordinates": [
        {
            "filename": "coordinate_sample1.jpg",
            "reference_image": "coordinate1.jpg",
            "option_images": ["jerry1.jpg", "jerry2.jpg", "jerry3.jpg", "jerry4.jpg", "jerry5.jpg"],
            "answer": {
                "choice": 4
            }
        },
        {
            "filename": "coordinate_sample2.jpg",
            "reference_image": "coordinate2.jpg",
            "option_images": ["jerry1.jpg", "jerry2.jpg", "jerry3.jpg", "jerry4.jpg", "jerry5.jpg"],
            "answer": {
                "choice": 3
            }
        }
    ],

    "Dart_Count": [
        {
            "filename": "dart_puzzle_1.json",
            "reference_image": "reference_num19.jpg",
            "option_images": ["1darts_1.jpg", "1darts_2.jpg", "2darts_1.jpg", "2darts_2.jpg", "2darts_3.jpg", "3darts_1.jpg", "3darts_2.jpg", "3darts_3.jpg", "3darts_4.jpg", "4darts_1.jpg", "4darts_2.jpg", "4darts_3.jpg"],
            "answer": {
                "choice": 10
            }
        },
        {
            "filename": "dart_puzzle_2.json",
            "reference_image": "reference_num30.jpg",
            "option_images": ["1darts_1.jpg", "1darts_2.jpg", "2darts_1.jpg", "2darts_2.jpg", "2darts_3.jpg", "3darts_1.jpg", "3darts_2.jpg", "3darts_3.jpg", "3darts_4.jpg", "4darts_1.jpg", "4darts_2.jpg", "4darts_3.jpg"],
            "answer": {
                "choice": 11
            }
        }
    ],

    "Dice_Count": [
        {
            "filename": "dice1.jpg",
            "answer": {
                "value": 85
            }
        },
        {
            "filename": "dice2.jpg",
            "answer": {
                "value": 67
            }
        }
    ],

    "Geometry_Click": [
        {
            "filename": "dingxiang_000001.jpg",
            "answer": {
                "point": {
                    "x": 0,
                    "y": 0
                }
            }
        },
        {
            "filename": "dingxiang_000002.jpg",
            "answer": {
                "point": {
                    "x": 0,
                    "y": 0
                }
            }
        }
    ],

    "Hold_Button(Not Used)": [
        {
            "filename": "image1.jpg",
            "answer": "completed"
        },
        {
            "filename": "image2.jpg",
            "answer": "completed"
        }
    ],

    "Image_Matching": [
        {
            "filename": "ground_image4.jpg",
            "reference_image": "ground_image4.jpg",
            "option_images": ["ground_image2_sub1.jpg", "ground_image2_sub2.jpg", "ground_image2_sub3.jpg", "ground_image2_sub6.jpg", "ground_image2_sub4.jpg", "ground_image2_sub5.jpg"],
            "answer": {
                "choice": 3
            }
        },
        {
            "filename": "ground_image2.jpg",
            "reference_image": "ground_image2.jpg",
            "option_images": ["ground_image2_sub1.jpg", "ground_image2_sub2.jpg", "ground_image2_sub3.jpg", "ground_image2_sub4.jpg", "ground_image2_sub5.jpg"],
            "answer": {
                "choice": 1
            }
        }
    ],

    "Image_Recognition": [
        {
            "filename": "images1"
        },
        {
            "filename": "images2"
        }
    ],

    "Misleading_Click": [
        {
            "filename": "image1.jpg",
            "answer": "avoid_red_bear"
        },
        {
            "filename": "image2.jpg",
            "answer": "avoid_person"
        }
    ],

    "Object_Match": [
        {
            "filename": "example1.jpg",
            "reference_image": "reference1.jpg",
            "option_images": ["image1.jpg", "image3.jpg", "image5.jpg", "image6.jpg", "image8.jpg"],
            "answer": {
                "choice": 0
            }
        },
        {
            "filename": "example2.jpg",
            "reference_image": "reference2.jpg",
            "option_images": ["image11.jpg", "image3.jpg", "image5.jpg", "image6.jpg", "image8.jpg"],
            "answer": {
                "choice": 0
            }
        }
    ],

    "Patch_Select": [
        {
            "filename": "image1.jpg",
            "target_object": "moon",
            "answer": {
                "indices": [0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 13]
            }
        },
        {
            "filename": "image2.jpg",
            "target_object": "butterfly",
            "answer": {
                "indices": [5, 6, 7, 10, 11, 12, 15, 16]
            }
        }
    ],

    "Path_Finder": [
        {
            "filename": "path1.JPG",
            "reference_image": "path1.JPG",
            "options": ["duck1.JPG", "duck2.JPG", "duck3.JPG", "duck4.JPG", "duck5.JPG"],
            "answer": {
                "choice": 2
            }
        },
        {
            "filename": "path2.JPG",
            "reference_image": "path2.JPG",
            "options": ["path2_img1.JPG", "path2_img2.JPG", "path2_img3.JPG", "path2_img4.JPG", "path2_img5.JPG"],
            "answer": {
                "choice": 4
            }
        }
    ],

    "Pick_Area": [
        {
            "filename": "image1.jpg",
            "answer": {
                "area": [[10, 150], [280, 490]],
                "type": "largest region"
            }
        },
        {
            "filename": "image2.jpg",
            "answer": {
                "area": [[200, 0], [500, 300]],
                "type": "largest region"
            }
        }
    ],

    "Place_Dot": [
        {
            "filename": "path1.jpg",
            "tolerance": 25,
            "answer": {
                "point": {
                    "x": 190,
                    "y": 210
                }
            }
        },
        {
            "filename": "path2.jpg",
            "tolerance": 15,
            "answer": {
                "point": {
                    "x": 265,
                    "y": 445
                }
            }
        }
    ],

    "Rotation_Match": [
        {
            "filename": "puzzle_cat_direction_1.json",
            "reference_image": "direction_1.jpg",
            "object_base_image": "cat.jpg",
            "answer": {
                "choice": 90
            }
        },
        {
            "filename": "puzzle_cat_direction_2.json",
            "reference_image": "direction_2.jpg",
            "object_base_image": "cat.jpg",
            "answer": {
                "choice": 45
            }
        }
    ],

    "Select_Animal": [
        {
            "filename": "image1.jpg"
        },
        {
            "filename": "image2.jpg"
        }
    ],

    "Slide_Puzzle(Not Used)": [
        {
            "filename": "slide2.jpg",
            "tolerance": 10,
            "answer": {
                "indices": [129, 68]
            }
        },
        {
            "filename": "slide3.jpg",
            "tolerance": 10,
            "answer": {
                "indices": [358, 62]
            }
        }
    ],

    "Unusual_Detection": [
        {
            "filename": "unusual1.jpg",
            "tolerance": 5,
            "answer": {
                "indices": [0, 3]
            }
        },
        {
            "filename": "unusual2.jpg",
            "tolerance": 5,
            "answer": {
                "indices": [0, 3]
            }
        }
    ]
}


def get_few_shot_answer(task_type: str, filename: str):

    if task_type not in FEW_SHOT_ANSWERS:
        return None

    examples = FEW_SHOT_ANSWERS[task_type]
    for example in examples:
        if example.get("filename") == filename:
            return example

    return None


def get_all_examples(task_type: str):

    return FEW_SHOT_ANSWERS.get(task_type, [])
