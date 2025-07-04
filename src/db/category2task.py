# src/db/category2task.py
# -------------------------------------
CATEGORY2TASK = {
    # ------ Vision 분류 ----------
    "Classification":        "classification",
    "Object Detection":      "object_detection",
    "Segmentation":          "segmentation",
    "Pose Estimation":       "pose_estimation",

    # ------ 복원·향상 -------------
    "Deblur":                "deblur",
    "Denoise":               "denoise",
    "Low Light Enhancement": "low_light_enhancement",
    "HDR":                   "hdr",
    "Weather Removal":       "weather_removal",
    "Restoration":           "restoration",

    # ------ 변환·생성 -------------
    "Colorization":          "colorization",
    "SISR":                  "super_resolution",
    "SISR Any":              "super_resolution",
    "Inpainting":            "inpainting",
    "ImgMsk2Img":            "inpainting",
    "Face Replacement":      "face_edit",
    "ImgTxt2Img":            "imgtxt2img",
    "Txt2Img":               "txt2img",
    "NST":                   "style_transfer",

    # ------ 텍스트·음성 -----------
    "Img2Txt":               "ocr",
    "Txt2Txt":               "text2text",
    "Txt2Voice":             "tts",
    "Voice2Txt":             "asr",
}