#  Realistic Shadow Generator

A shadow compositing tool available in **both Python CLI and TypeScript/Web** versions, demonstrating realistic directional shadows with contact shadows and soft falloff.

##  Challenge Requirements Met

 **Directional light control** - Angle (0-360°) and elevation (0-90°)  
 **Contact shadow** - Dark/sharp near base, fades quickly  
 **Soft shadow falloff** - Blur increases + opacity decreases with distance  
 **Silhouette-based shadow** - Matches exact subject shape (not oval/drop-shadow)  
 **Depth map support** - Warps shadow based on surface depth (Python version)  
 **Python CLI** - Full-featured command-line tool  
 **TypeScript/Web App** - Browser-based interactive UI  

---

##  Web Version (TypeScript/HTML)

Open `index.html` in any browser - no build tools required!

### Features
- Drag & drop image uploads
- Real-time shadow preview
- Interactive light direction control with visual indicator
- Download composite, shadow, and mask as PNG
- Responsive dark UI

### Usage
1. Open `index.html` in a browser
2. Upload a foreground image (subject)
3. Upload a background image
4. Adjust light angle, elevation, and shadow settings
5. Download the result

---

##  Python CLI Version

## Features

###  Core Requirements
- ** Directional Light Control**
  - Light angle (0–360°): Controls shadow direction
  - Light elevation (0–90°): Controls shadow length (higher = shorter shadow)

- ** Contact Shadow**
  - Dark and sharp near the contact area
  - Quickly fades with distance

- ** Soft Shadow Falloff**
  - Blur increases as shadow extends
  - Opacity decreases with distance

- ** Silhouette-Based Shadow**
  - Shadow matches the exact subject shape
  - Automatic background removal with rembg

###  Bonus: Depth Map Support
When a depth map is provided, the shadow warps based on surface depth for more realistic behavior on uneven surfaces.

## Installation

```bash
pip install opencv-python numpy pillow rembg
```

## Usage

### Basic Usage
```bash
python shadow_generator.py -f foreground.jpg -b background.jpg
```

### With Custom Light Direction
```bash
python shadow_generator.py -f subject.png -b background.jpg --angle 45 --elevation 30

python shadow_generator.py -f subject.png -b background.jpg --angle 135 --elevation 45
```

### With Depth Map (Bonus Mode)
```bash
python shadow_generator.py -f subject.png -b background.jpg -d depth.png
```

### All Options
```bash
python shadow_generator.py \
  -f foreground.jpg \
  -b background.jpg \
  -d depth.png \
  -o output/ \
  --angle 135 \
  --elevation 45 \
  --shadow-length 150 \
  --contact-darkness 0.9 \
  --scale 0.8 \
  --pos-x 100 \
  --pos-y 200
```


## Output Files

| File | Description |
|------|-------------|
| `composite.png` | Final composited image |
| `shadow_only.png` | Debug view of shadow only |
| `mask_debug.png` | Debug view of subject cutout mask |

## How It Works

### Shadow Generation Algorithm

1. **Background Removal**: Uses `rembg` to extract subject silhouette
2. **Shadow Layers**: Creates multiple offset copies of the mask at varying distances
3. **Progressive Blur**: Each layer has increasing Gaussian blur (sharp near contact, soft at distance)
4. **Opacity Falloff**: Each layer has decreasing opacity
5. **Contact Shadow**: Additional sharp, dark shadow right at the subject base
6. **Depth Warping** (optional): Uses depth map gradients to displace shadow realistically

### Light Direction Math

```
shadow_direction = light_angle + 180° 
shadow_length = max_length × cos(elevation)  
```

## Examples

### Child on Classroom Background
```bash
python shadow_generator.py \
  -f child_photo.jpg \
  -b classroom.jpg \
  --angle 120 \
  --elevation 40
```

### Car on Street
```bash
python shadow_generator.py \
  -f car.jpg \
  -b street.jpg \
  --angle 90 \
  --elevation 60 \
  --shadow-length 200
```

