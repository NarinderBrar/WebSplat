export type EditorTool =
  | "orbit"
  | "brushSelect"
  | "paintBrush"
  | "marqueeSelect"
  | "lassoSelect"
  | "hsv"
  | "colorize"
  | "hide"
  | "move"
  | "rotate"
  | "scale";

export type SelectionMode = "normal" | "additive" | "subtractive";

export interface ScreenPoint {
  x: number;
  y: number;
}

export interface ScreenRect {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface HsvAdjust {
  hue: number;
  saturation: number;
  value: number;
}
