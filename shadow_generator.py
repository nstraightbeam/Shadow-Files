#!/usr/bin/env python3
"""
Realistic Shadow Generator
===========================
A Python CLI tool that composites foreground subjects onto backgrounds
with realistic, directional shadows including contact shadows and soft falloff.

Author: Yixin Zhang
"""

import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import math


def remove_background(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove background from foreground image using combined techniques:
    - Edge detection
    - Color-based segmentation
    - GrabCut refinement
    Returns the RGBA image and the alpha mask.
    """
    h, w = image.shape[:2]
    

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
  
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    

    corner_size = 20
    corners = [
        image[:corner_size, :corner_size],
        image[:corner_size, -corner_size:],
        image[-corner_size:, :corner_size],
        image[-corner_size:, -corner_size:]
    ]
    bg_colors = np.vstack([c.reshape(-1, 3) for c in corners])
    bg_mean = np.mean(bg_colors, axis=0)
    bg_std = np.std(bg_colors, axis=0)
    
    diff = np.abs(image.astype(np.float32) - bg_mean)
    color_dist = np.sqrt(np.sum(diff ** 2, axis=2))

    threshold = np.mean(color_dist) + np.std(color_dist) * 0.5
    fg_mask = (color_dist > threshold).astype(np.uint8) * 255
    
    combined = cv2.bitwise_or(fg_mask, edges)
    
    kernel = np.ones((7, 7), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        mask = fg_mask
    
    if np.sum(mask > 0) > 1000:  
        try:
            gc_mask = np.zeros((h, w), np.uint8)
            gc_mask[mask > 0] = cv2.GC_PR_FGD
            gc_mask[mask == 0] = cv2.GC_PR_BGD
            
            center_y, center_x = h // 2, w // 2
            margin = min(h, w) // 4
            gc_mask[center_y-margin:center_y+margin, center_x-margin:center_x+margin] = cv2.GC_FGD
            
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(image, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            
            refined_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype('uint8')
            
            if np.sum(refined_mask > 0) > 1000:
                mask = refined_mask
        except:
            pass  

    kernel_small = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    

    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask
    
    return rgba, mask


def create_shadow_from_mask(
    mask: np.ndarray,
    light_angle: float,
    light_elevation: float,
    shadow_length: float = 100,
    contact_darkness: float = 0.8,
    contact_size: int = 5,
    falloff_blur_start: int = 3,
    falloff_blur_end: int = 50,
    falloff_opacity_start: float = 0.7,
    falloff_opacity_end: float = 0.0,
) -> np.ndarray:
    """
    Create a realistic shadow from a silhouette mask.
    
    Args:
        mask: Binary mask of the subject (255 = subject, 0 = background)
        light_angle: Horizontal angle of light source (0-360°, 0=right, 90=top, 180=left, 270=bottom)
        light_elevation: Vertical angle of light source (0-90°, 0=horizon, 90=directly above)
        shadow_length: Maximum shadow length in pixels
        contact_darkness: Darkness of contact shadow (0-1)
        contact_size: Size of contact shadow blur
        falloff_blur_start: Initial blur amount near contact
        falloff_blur_end: Final blur amount at shadow end
        falloff_opacity_start: Initial opacity near contact
        falloff_opacity_end: Final opacity at shadow end
    
    Returns:
        Shadow image (grayscale, 0=shadow, 255=no shadow)
    """
    h, w = mask.shape
    

    shadow_angle_rad = math.radians(light_angle + 180)
    
    elevation_factor = math.cos(math.radians(light_elevation))
    actual_shadow_length = shadow_length * elevation_factor
    
    dx = math.cos(shadow_angle_rad)
    dy = -math.sin(shadow_angle_rad)  
    

    shadow = np.zeros((h, w), dtype=np.float32)
    
    num_layers = max(1, int(actual_shadow_length / 2))
    
    for i in range(num_layers):
       
        t = i / max(1, num_layers - 1)  
        
        offset_x = int(dx * actual_shadow_length * t)
        offset_y = int(dy * actual_shadow_length * t)
        
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        shifted_mask = cv2.warpAffine(mask.astype(np.float32), M, (w, h))

        blur_amount = int(falloff_blur_start + (falloff_blur_end - falloff_blur_start) * t)
        if blur_amount > 0:
            blur_amount = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
            shifted_mask = cv2.GaussianBlur(shifted_mask, (blur_amount, blur_amount), 0)

        layer_opacity = falloff_opacity_start + (falloff_opacity_end - falloff_opacity_start) * t

        shadow = np.maximum(shadow, shifted_mask * layer_opacity)
    

    contact_shadow = np.zeros((h, w), dtype=np.float32)

    kernel = np.ones((contact_size, contact_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)

    contact_offset_x = int(dx * contact_size * 2)
    contact_offset_y = int(dy * contact_size * 2)
    M_contact = np.float32([[1, 0, contact_offset_x], [0, 1, contact_offset_y]])
    contact_mask = cv2.warpAffine(dilated.astype(np.float32), M_contact, (w, h))
    
    contact_blur = contact_size * 2 + 1
    contact_shadow = cv2.GaussianBlur(contact_mask, (contact_blur, contact_blur), 0)
    contact_shadow = contact_shadow * contact_darkness
    
    combined_shadow = np.maximum(shadow, contact_shadow)
    
    combined_shadow = np.clip(combined_shadow, 0, 255)

    shadow_intensity = (combined_shadow / 255.0)
    
    return shadow_intensity


def apply_depth_warp(
    shadow: np.ndarray,
    depth_map: np.ndarray,
    warp_strength: float = 0.3
) -> np.ndarray:
    """
    Warp shadow based on depth map for more realistic shadows on uneven surfaces.
    
    Args:
        shadow: Shadow intensity map
        depth_map: Grayscale depth map (0=far, 255=near)
        warp_strength: How much the depth affects shadow warping
    
    Returns:
        Warped shadow
    """
    h, w = shadow.shape
    
    if depth_map.shape[:2] != shadow.shape:
        depth_map = cv2.resize(depth_map, (w, h))
    
    if len(depth_map.shape) == 3:
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    depth_norm = depth_map.astype(np.float32) / 255.0
    
    gradient_x = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=5)
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    
    displacement = warp_strength * 50
    map_x = x + gradient_x * displacement
    map_y = y + gradient_y * displacement
    
    warped = cv2.remap(shadow, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    depth_factor = 0.5 + 0.5 * depth_norm  
    warped = warped * depth_factor
    
    return warped


def composite_with_shadow(
    foreground_rgba: np.ndarray,
    background: np.ndarray,
    shadow: np.ndarray,
    position: Tuple[int, int] = (0, 0),
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
    scale: float = 1.0
) -> np.ndarray:
    """
    Composite foreground onto background with shadow.
    
    Args:
        foreground_rgba: RGBA foreground image
        background: BGR background image
        shadow: Shadow intensity map (0-1, 1 = full shadow)
        position: (x, y) position to place foreground
        shadow_color: RGB color of shadow
        scale: Scale factor for foreground
    
    Returns:
        Composited BGR image
    """
    bg_h, bg_w = background.shape[:2]
    
    if scale != 1.0:
        new_w = int(foreground_rgba.shape[1] * scale)
        new_h = int(foreground_rgba.shape[0] * scale)
        foreground_rgba = cv2.resize(foreground_rgba, (new_w, new_h))
        shadow = cv2.resize(shadow, (new_w, new_h))
    
    fg_h, fg_w = foreground_rgba.shape[:2]
    
    result = background.copy().astype(np.float32)

    x, y = position
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + fg_w), min(bg_h, y + fg_h)
    
    fx1, fy1 = max(0, -x), max(0, -y)
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return background
    
    bg_region = result[y1:y2, x1:x2]
    fg_region = foreground_rgba[fy1:fy2, fx1:fx2]
    shadow_region = shadow[fy1:fy2, fx1:fx2]
    
    shadow_3ch = np.stack([shadow_region] * 3, axis=-1)
    shadow_color_np = np.array(shadow_color, dtype=np.float32).reshape(1, 1, 3)
    
    bg_with_shadow = bg_region * (1 - shadow_3ch * 0.6) + shadow_color_np * shadow_3ch * 0.6
    
    if fg_region.shape[2] == 4:
        alpha = fg_region[:, :, 3:4].astype(np.float32) / 255.0
        fg_rgb = fg_region[:, :, :3].astype(np.float32)
        fg_bgr = fg_rgb[:, :, ::-1]
        
        blended = bg_with_shadow * (1 - alpha) + fg_bgr * alpha
        result[y1:y2, x1:x2] = blended
    
    return np.clip(result, 0, 255).astype(np.uint8)


def create_shadow_only_image(shadow: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Create a debug image showing only the shadow."""
    h, w = size
    shadow_resized = cv2.resize(shadow, (w, h)) if shadow.shape[:2] != (h, w) else shadow
    shadow_vis = (shadow_resized * 255).astype(np.uint8)
    shadow_vis = 255 - shadow_vis  
    return cv2.cvtColor(shadow_vis, cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(
        description="Realistic Shadow Generator - Composite foreground onto background with realistic shadows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python shadow_generator.py -f subject.png -b background.jpg -o output/
  python shadow_generator.py -f subject.png -b background.jpg --angle 45 --elevation 30
  python shadow_generator.py -f subject.png -b background.jpg -d depth.png --angle 135
        """
    )
    
    parser.add_argument("-f", "--foreground", required=True, help="Foreground image (subject will be cut out)")
    parser.add_argument("-b", "--background", required=True, help="Background image")
    parser.add_argument("-d", "--depth", help="Optional depth map for advanced shadow warping")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    
    parser.add_argument("--angle", type=float, default=135, help="Light angle 0-360° (default: 135, upper-left)")
    parser.add_argument("--elevation", type=float, default=45, help="Light elevation 0-90° (default: 45)")
    
    parser.add_argument("--shadow-length", type=float, default=150, help="Max shadow length in pixels (default: 150)")
    parser.add_argument("--contact-darkness", type=float, default=0.9, help="Contact shadow darkness 0-1 (default: 0.9)")
    parser.add_argument("--scale", type=float, default=0.8, help="Scale factor for foreground (default: 0.8)")
    parser.add_argument("--pos-x", type=int, default=None, help="X position (default: center)")
    parser.add_argument("--pos-y", type=int, default=None, help="Y position (default: bottom-aligned)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🖼️  Loading foreground: {args.foreground}")
    foreground = cv2.imread(args.foreground)
    if foreground is None:
        raise FileNotFoundError(f"Could not load foreground image: {args.foreground}")
    
    print(f"🏞️  Loading background: {args.background}")
    background = cv2.imread(args.background)
    if background is None:
        raise FileNotFoundError(f"Could not load background image: {args.background}")
    
    depth_map = None
    if args.depth:
        print(f"🌫️  Loading depth map: {args.depth}")
        depth_map = cv2.imread(args.depth, cv2.IMREAD_GRAYSCALE)
    
    print("✂️  Removing background from foreground...")
    foreground_rgba, mask = remove_background(foreground)
    
    mask_debug_path = output_dir / "mask_debug.png"
    cv2.imwrite(str(mask_debug_path), mask)
    print(f"✅ Saved mask debug: {mask_debug_path}")
    
    print(f"🖤 Generating shadow (angle={args.angle}°, elevation={args.elevation}°)...")
    shadow = create_shadow_from_mask(
        mask=mask,
        light_angle=args.angle,
        light_elevation=args.elevation,
        shadow_length=args.shadow_length,
        contact_darkness=args.contact_darkness,
    )

    if depth_map is not None:
        print("🌫️  Applying depth-based shadow warping...")
        shadow = apply_depth_warp(shadow, depth_map)
    
    shadow_debug = create_shadow_only_image(shadow, (foreground.shape[1], foreground.shape[0]))
    shadow_only_path = output_dir / "shadow_only.png"
    cv2.imwrite(str(shadow_only_path), shadow_debug)
    print(f"✅ Saved shadow debug: {shadow_only_path}")
    
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = int(foreground_rgba.shape[0] * args.scale), int(foreground_rgba.shape[1] * args.scale)
    
    pos_x = args.pos_x if args.pos_x is not None else (bg_w - fg_w) // 2
    pos_y = args.pos_y if args.pos_y is not None else bg_h - fg_h - 20  # Align to bottom with padding
    
    print("🎨 Compositing final image...")
    result = composite_with_shadow(
        foreground_rgba=foreground_rgba,
        background=background,
        shadow=shadow,
        position=(pos_x, pos_y),
        scale=args.scale,
    )
    
    composite_path = output_dir / "composite.png"
    cv2.imwrite(str(composite_path), result)
    print(f"✅ Saved composite: {composite_path}")
    
    print("\n🎉 Done! Generated files:")
    print(f"   📄 {composite_path}")
    print(f"   📄 {shadow_only_path}")
    print(f"   📄 {mask_debug_path}")


if __name__ == "__main__":
    main()
