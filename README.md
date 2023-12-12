# cs1290_fp

Mehek and Noah Final Project: Impressionist Paintings

## Run the code

`usage: main.py [-h] [-m MASK] [-l LENGTH] [-r RADIUS] [-a ANGLE] [-p] [-c] [-o] [-i] [-v] [-f] input`

```
positional arguments:
  input                         Input img/video filepath

options:
  -h, --help                    Show help message with these args
  -m MASK, --mask MASK          Brush mask filepath
  -l LENGTH, --length LENGTH    Brush stroke length (int)
  -r RADIUS, --radius RADIUS    Brush radius (int)
  -a ANGLE, --angle ANGLE       Brush angle (int)
  -p, --perturb                 Don't randomly perturb stroke colors and angles
  -c, --clip                    Don't clip strokes at edges
  -o, --orient                  Don't orient strokes based on gradients
  -i, --interp                  Don't interpolate stroke gradient directions
  -f, --flow                    Don't use optical flow to optimize painting video frames
  -v, --view                    Show view of stroke placement in real time
```
