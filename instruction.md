# Camera / Video Input Instructions for Monocular 3D Reconstruction

This project is a monocular 3D reconstruction pipeline. It expects a short RGB video of a static object captured while the camera moves around it. The system uses ORB feature tracking, pose estimation, and relative depth estimation to reconstruct a coarse 3D scene. The results are in arbitrary relative units, not real metric units.

## 1. What kind of video works best

The pipeline works best when the input video contains:

- A single static object or scene in the center of the frame
- Rich texture and visible corners/edges
- Good lighting and strong contrast
- Smooth camera motion around the subject
- A stable, continuous sequence of frames

The camera should move around the object, not just passively look at it. The code expects parallax: motion across multiple frames.

## 2. Best scene setup

Use:

- A textured object such as a chair, backpack, statue, rock, toy, box, or product item
- A clean indoor background or simple table scene
- Objects with visible edges, corners, fabric, wood grain, stone texture, or patterned surfaces
- Soft but strong lighting

Avoid:

- Plain white walls
- Reflective glass or mirrors
- Smooth shiny plastic or metal surfaces
- Featureless objects
- Strong motion blur
- People or moving objects in the frame
- Repetitive patterns with too little variation

## 3. Camera motion requirements

The video should show the camera slowly moving in an arc around the object while keeping the object centered.

Recommended motion pattern:

- Start 1 to 2 meters away from the object
- Move around the object in a slow arc or half-circle
- Keep the object in the center of the frame
- Keep the motion smooth and continuous
- Avoid sudden jerks, hand tremors, or rapid zooms

## 4. Recommended video specifications

- Resolution: 720p or 1080p preferred
- Frame rate: 24–30 fps
- Duration: 5–15 seconds
- Camera: smartphone or consumer camera
- Focus: sharp, stable focus
- Lighting: well-lit, no extreme shadows or overexposure

## 5. Good input description for generation

Use this as a text prompt for generating a source video or reference visual:

"Generate a 10-second handheld smartphone video of a static textured object centered on a table in a well-lit indoor room. The camera slowly moves in a smooth arc around the object from 1 to 2 meters away, keeping the object centered in frame throughout. The object has rich visible features such as edges, corners, fabric texture, rough materials, wood grain, or patterned surfaces. No reflective glass, mirrors, plain white walls, or featureless surfaces. The motion is steady and continuous, with no sudden shakes, zooms, or blurry frames. Realistic lighting, 24–30 fps, 1080p quality, no people, no moving objects, suitable for monocular 3D reconstruction."

## 6. Good input description for reference image generation

"Create a realistic product-shot still image of a textured, non-reflective object placed on a table in a clean indoor scene. The object should have strong visible edges, corners, rough or patterned surfaces, and high contrast details. It should be centered in the frame and well lit with soft daylight or studio lighting. Avoid glossy, plain-white, or low-texture surfaces. The object should look like a good subject for 3D reconstruction from a moving camera."

## 7. What the algorithm expects internally

This pipeline is designed around:

- ORB feature detection
- Matching between consecutive frames
- Essential matrix estimation using RANSAC
- Camera pose recovery from matched points
- Relative depth estimation from a model such as Depth Anything V2
- Backprojection of depth into 3D points
- Multi-view point cloud fusion and meshing

Because of this, the best input has:

- enough corners and textures for feature matching
- enough parallax for pose estimation
- enough object visibility across frames
- enough depth variation in the scene

## 8. Red flags to avoid

Do not use videos where:

- The camera is mostly static
- The object is too far away or too small
- The subject is monotone or textureless
- The camera pans too fast
- The scene is extremely dark or blurry
- The object is reflective or flat
- The object is moving while the camera also moves

## 9. Summary

For this codebase, the ideal input is not a random video clip. It is a short, smooth orbit around a textured object with strong visual features. The object must remain centered, the motion must be continuous, and the footage must contain clear depth cues and stable tracked points.

This project is specifically built for monocular motion-based reconstruction, so focus on motion + texture + stable framing.
