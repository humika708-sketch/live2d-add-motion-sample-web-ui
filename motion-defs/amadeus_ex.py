"""Motion definitions for Amadeus Kurisu (amadeus_ex).

The model ships without any motions, so the validator cannot derive safe
ranges or a base pose from existing curves. Ranges and defaults below were
read from the Cubism Editor project (amadeus_ex_recovered.cmo3, main.xml):

  ParamEyeROpen    0..1   default 1   (drives BOTH eyes despite the "R")
  ParamEyeRSmile   0..1   default 0   (both eyes; needs EyeOpen=0 to read)
  ParamEyeBallX/Y -1..1   default 0
  ParamMouthOpenY  0..1   default 0
  ParamMouthForm  -1..1   default 0
  Param6           0..10  default 0   Arms Thinking Movement (hand to chin;
                                      keyforms at 0/1/1.9/3.4/4/10)
  Param8           0..1   default 0   Thinking FaceExpression (mouth + cheeks)
  Param9           0..1   default 0   Blushing

Physics outputs (never animated here): Param, Param2, Param3 (eye lashes /
iris, from EyeOpen), Param4, Param5 (arm / hair, from Breath), Param7 (arm
follow-through, from Param6). ParamBreath is left to the runtime's
automatic breathing.

This rig has no head/body angle parameters and no brows, so every motion is
expressed through eyes, gaze, mouth, blush and the thinking-arm pose.
"""


def define(curve, motion):
    MOTIONS = {}

    # ------------------------------------------------------------ thinking
    # Hand to chin, eyes drift up and aside, then back.
    D = 4.5
    MOTIONS["kurisu_thinking"] = motion(D, [
        curve("Param6", [(0, 0), (1.2, 10), (3.4, 10), (D, 0)]),
        curve("Param8", [(0, 0), (0.9, 1), (3.4, 1), (4.2, 0), (D, 0)]),
        curve("ParamEyeBallX", [(0, 0), (0.9, -0.6), (2.2, -0.7), (2.6, -0.3), (3.4, -0.6), (4.1, 0), (D, 0)]),
        curve("ParamEyeBallY", [(0, 0), (0.9, 0.6), (3.4, 0.6), (4.1, 0), (D, 0)]),
        curve("ParamEyeROpen", [(0, 1), (0.9, 0.75), (2.3, 0.75), (2.45, 0), (2.6, 0.75), (3.4, 0.75), (4.1, 1), (D, 1)]),
        curve("ParamMouthForm", [(0, 0), (0.9, -0.4), (3.4, -0.4), (4.1, 0), (D, 0)]),
    ])

    # ------------------------------------------------------------ smile
    # Soft closed-eye smile (^^) with a light blush.
    D = 3.0
    MOTIONS["kurisu_smile"] = motion(D, [
        curve("ParamEyeROpen", [(0, 1), (0.35, 0), (2.3, 0), (2.7, 1), (D, 1)]),
        curve("ParamEyeRSmile", [(0, 0), (0.35, 1), (2.3, 1), (2.7, 0), (D, 0)]),
        curve("ParamMouthForm", [(0, 0), (0.35, 1), (2.3, 1), (2.8, 0), (D, 0)]),
        curve("ParamMouthOpenY", [(0, 0), (0.4, 0.35), (1.2, 0.2), (2.2, 0.25), (2.7, 0), (D, 0)]),
        curve("Param9", [(0, 0), (0.5, 0.35), (2.3, 0.35), (2.9, 0), (D, 0)]),
    ])

    # ------------------------------------------------------------ tsundere
    # "It's not like I'm happy or anything!" Full blush, eyes averted, pout.
    D = 4.0
    MOTIONS["kurisu_tsundere"] = motion(D, [
        curve("Param9", [(0, 0), (0.5, 1), (3.3, 1), (D, 0)]),
        curve("ParamEyeBallX", [(0, 0), (0.4, 0.8), (1.8, 0.8), (2.1, 0.2), (2.4, 0.8), (3.3, 0.8), (3.8, 0), (D, 0)]),
        curve("ParamEyeBallY", [(0, 0), (0.4, -0.4), (3.3, -0.4), (3.8, 0), (D, 0)]),
        curve("ParamEyeROpen", [(0, 1), (0.4, 0.7), (3.3, 0.7), (3.8, 1), (D, 1)]),
        curve("ParamMouthForm", [(0, 0), (0.4, -0.7), (3.3, -0.7), (3.8, 0), (D, 0)]),
        curve("ParamMouthOpenY", [(0, 0), (0.5, 0.1), (1.0, 0.25), (1.3, 0.05), (1.6, 0.2), (2.0, 0), (D, 0)]),
    ])

    # ------------------------------------------------------------ surprised
    # No brows or head angles: surprise reads from a startled blink,
    # a wide-open mouth and pupils snapping to centre.
    D = 2.5
    MOTIONS["kurisu_surprised"] = motion(D, [
        curve("ParamEyeROpen", [(0, 1), (0.08, 0.2), (0.2, 1), (D, 1)]),
        curve("ParamEyeBallY", [(0, 0), (0.2, 0.25), (1.5, 0.25), (2.1, 0), (D, 0)]),
        curve("ParamMouthOpenY", [(0, 0), (0.2, 0.9), (1.4, 0.8), (2.1, 0), (D, 0)]),
        curve("ParamMouthForm", [(0, 0), (0.2, -1), (1.4, -1), (2.1, 0), (D, 0)]),
    ])

    # ------------------------------------------------------------ look around
    # Glancing left, right and up with a blink between.
    D = 4.0
    MOTIONS["kurisu_lookaround"] = motion(D, [
        curve("ParamEyeBallX", [(0, 0), (0.4, -0.9), (1.2, -0.9), (1.6, 0.9), (2.4, 0.9), (2.8, 0), (3.6, 0), (D, 0)]),
        curve("ParamEyeBallY", [(0, 0), (1.2, 0), (1.6, 0.1), (2.4, 0.1), (2.8, 0.7), (3.4, 0.7), (3.8, 0), (D, 0)]),
        curve("ParamEyeROpen", [(0, 1), (1.3, 1), (1.4, 0), (1.5, 1), (D, 1)]),
        curve("ParamMouthForm", [(0, 0), (0.4, -0.2), (3.4, -0.2), (3.8, 0), (D, 0)]),
    ])

    # ------------------------------------------------------------ explaining
    # Lecturing mode: talking mouth flaps, a confident half-smile, a blink.
    D = 4.0
    MOTIONS["kurisu_explain"] = motion(D, [
        curve("ParamMouthOpenY", [(0, 0), (0.2, 0.6), (0.4, 0.15), (0.6, 0.7), (0.85, 0.2), (1.05, 0.5),
                                  (1.3, 0.1), (1.6, 0.1), (1.8, 0.6), (2.0, 0.2), (2.25, 0.75), (2.5, 0.25),
                                  (2.7, 0.5), (2.95, 0.1), (3.2, 0.4), (3.5, 0), (D, 0)]),
        curve("ParamMouthForm", [(0, 0), (0.3, 0.4), (1.4, 0.2), (2.2, 0.6), (3.4, 0.4), (3.9, 0), (D, 0)]),
        curve("ParamEyeBallX", [(0, 0), (0.5, 0.3), (1.4, 0.3), (1.8, -0.2), (3.0, -0.2), (3.6, 0), (D, 0)]),
        curve("ParamEyeROpen", [(0, 1), (1.45, 1), (1.55, 0), (1.65, 1), (D, 1)]),
        curve("ParamEyeRSmile", [(0, 0), (D, 0)]),
    ])

    # ------------------------------------------------------------ sleepy
    # Eyelids droop, a yawn, then snapping awake.
    D = 5.0
    MOTIONS["kurisu_sleepy"] = motion(D, [
        curve("ParamEyeROpen", [(0, 1), (1.0, 0.45), (1.5, 0.55), (2.4, 0.1), (2.9, 0.1),
                                (3.8, 0.1), (3.95, 1), (D, 1)]),
        curve("ParamEyeBallY", [(0, 0), (1.0, -0.3), (3.8, -0.3), (3.95, 0), (D, 0)]),
        curve("ParamMouthOpenY", [(0, 0), (1.5, 0), (2.2, 1), (2.9, 1), (3.4, 0), (D, 0)]),
        curve("ParamMouthForm", [(0, 0), (1.5, 0), (2.2, -0.6), (2.9, -0.6), (3.4, 0), (D, 0)]),
    ])

    # ------------------------------------------------------------ eureka
    # Chin-in-hand pondering, then the idea lands: eyes wide, bright smile.
    D = 5.0
    MOTIONS["kurisu_eureka"] = motion(D, [
        curve("Param6", [(0, 0), (1.0, 10), (2.2, 10), (3.0, 0), (D, 0)]),
        curve("Param8", [(0, 0), (0.8, 1), (2.2, 1), (2.5, 0), (D, 0)]),
        curve("ParamEyeBallX", [(0, 0), (0.8, 0.5), (2.2, 0.5), (2.4, 0), (D, 0)]),
        curve("ParamEyeBallY", [(0, 0), (0.8, 0.5), (2.2, 0.5), (2.4, 0), (D, 0)]),
        curve("ParamEyeROpen", [(0, 1), (0.8, 0.6), (2.2, 0.6), (2.35, 1), (3.3, 1), (3.6, 0), (4.4, 0), (4.8, 1), (D, 1)]),
        curve("ParamEyeRSmile", [(0, 0), (3.3, 0), (3.6, 1), (4.4, 1), (4.8, 0), (D, 0)]),
        curve("ParamMouthForm", [(0, 0), (0.8, -0.3), (2.2, -0.3), (2.4, 0.8), (4.4, 1), (4.9, 0), (D, 0)]),
        curve("ParamMouthOpenY", [(0, 0), (2.2, 0), (2.4, 0.7), (3.0, 0.5), (3.6, 0.4), (4.4, 0.2), (4.8, 0), (D, 0)]),
        curve("Param9", [(0, 0), (3.3, 0), (3.7, 0.3), (4.4, 0.3), (4.9, 0), (D, 0)]),
    ])

    # Entries registered into the "Action" group of model3.json
    # (display name shown in the WebUI, fade-in/out seconds)
    MANIFEST = [
        ("kurisu_thinking", "考え中", 0.5, 0.5),
        ("kurisu_smile", "微笑み", 0.4, 0.5),
        ("kurisu_tsundere", "ツンデレ", 0.4, 0.5),
        ("kurisu_surprised", "びっくり", 0.2, 0.5),
        ("kurisu_lookaround", "きょろきょろ", 0.4, 0.5),
        ("kurisu_explain", "解説モード", 0.3, 0.5),
        ("kurisu_sleepy", "眠い", 0.5, 0.5),
        ("kurisu_eureka", "ひらめき", 0.5, 0.5),
    ]
    return MOTIONS, MANIFEST
