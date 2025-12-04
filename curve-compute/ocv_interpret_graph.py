import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
IMAGE_PATH = 'image_912f6a.png'
OUTPUT_CSV = 'red_curve_data.csv'

# Graph Axis Values (Based on the labels in your image)
# Update these if you process a different graph
X_AXIS_MAX_VALUE = 4000.0  # mAh
Y_AXIS_MAX_VALUE = 5.0     # Volts

class GraphDigitizer:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        self.clone = self.img.copy()
        self.points = []
        self.ref_pts = {} # Stores 'origin', 'x_max', 'y_max'

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Visual feedback
            cv2.circle(self.clone, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Calibrate Graph", self.clone)
            self.points.append((x, y))

            # Logic for calibration steps
            if len(self.points) == 1:
                print(f"Origin set at: {self.points[0]}")
            elif len(self.points) == 2:
                print(f"X-Axis Max set at: {self.points[1]}")
            elif len(self.points) == 3:
                print(f"Y-Axis Max set at: {self.points[2]}")
                print("Calibration complete. Press any key to continue...")

    def calibrate(self):
        """
        Interactive function to let the user define the axes.
        """
        print("\n--- CALIBRATION REQUIRED ---")
        print("Please click the following points on the popup window IN ORDER:")
        print("1. The ORIGIN (0, 0) - Bottom Left corner of the graph axes.")
        print("2. The MAX X point (4000, 0) - Bottom Right tick mark.")
        print("3. The MAX Y point (0, 5.0) - Top Left tick mark.")
        print("----------------------------")
        
        cv2.namedWindow("Calibrate Graph")
        cv2.setMouseCallback("Calibrate Graph", self.mouse_callback)
        cv2.imshow("Calibrate Graph", self.img)
        
        # Wait until 3 points are clicked or ESC is pressed
        while len(self.points) < 3:
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC
                print("Calibration cancelled.")
                cv2.destroyAllWindows()
                return False

        cv2.waitKey(0) # Wait for a final key press
        cv2.destroyAllWindows()

        self.ref_pts['origin'] = self.points[0]
        self.ref_pts['x_max'] = self.points[1]
        self.ref_pts['y_max'] = self.points[2]
        return True

    def get_pixel_to_unit_ratios(self):
        """Calculates scale factors based on calibrated points."""
        origin = self.ref_pts['origin']
        x_pt = self.ref_pts['x_max']
        y_pt = self.ref_pts['y_max']

        # Pixel distances
        x_dist_px = x_pt[0] - origin[0]
        y_dist_px = origin[1] - y_pt[1] # Y is inverted in images (0 is top)

        # Ratios (Units per Pixel)
        x_scale = X_AXIS_MAX_VALUE / x_dist_px
        y_scale = Y_AXIS_MAX_VALUE / y_dist_px

        return x_scale, y_scale

    def extract_red_curve(self):
        """Isolates the red curve and filters out noise."""
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # Define red ranges (Red wraps around 180 in HSV)
        # Lower mask (0-10)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        
        # Upper mask (170-180)
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Noise removal
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Filter: Keep only the largest connected component (the main curve)
        # This removes the small red line in the legend
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # stats columns: [x, y, width, height, area]
        # We skip label 0 (background)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_clean = np.zeros_like(mask)
            mask_clean[labels == largest_label] = 255
        else:
            mask_clean = mask

        return mask_clean

    def process(self):
        if not self.calibrate():
            return

        mask = self.extract_red_curve()
        
        # Visual check of the mask
        cv2.imshow("Extracted Curve (Mask)", mask)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

        x_scale, y_scale = self.get_pixel_to_unit_ratios()
        origin = self.ref_pts['origin']

        data_points = []

        # Iterate through columns (X pixels) to find the curve height (Y)
        # We start from the origin X and move right
        height, width = mask.shape
        
        # Optimization: Use numpy to find all non-zero pixels
        ys, xs = np.nonzero(mask)
        
        # Create a dataframe to group by X and average Y
        df_pixels = pd.DataFrame({'px_x': xs, 'px_y': ys})
        
        # Group by X to handle line thickness (take the mean Y for every X)
        curve_pixels = df_pixels.groupby('px_x')['px_y'].mean().reset_index()
        
        # Sort by X
        curve_pixels = curve_pixels.sort_values('px_x')

        results = []
        for index, row in curve_pixels.iterrows():
            px_x = row['px_x']
            px_y = row['px_y']

            # 1. Shift coordinates to be relative to origin
            rel_x = px_x - origin[0]
            rel_y = origin[1] - px_y # Inverted Y axis

            # 2. Filter out points that are "behind" the Y-axis (noise to the left)
            if rel_x < 0:
                continue

            # 3. Convert to units
            capacity_mah = rel_x * x_scale
            voltage_v = rel_y * y_scale

            results.append((capacity_mah, voltage_v))

        # Create DataFrame
        df_final = pd.DataFrame(results, columns=['Capacity (mAh)', 'Voltage (V)'])
        
        # Save
        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Extracted {len(df_final)} data points.")
        print(f"Data saved to {OUTPUT_CSV}")

        # Verification Plot
        plt.figure(figsize=(10, 6))
        plt.plot(df_final['Capacity (mAh)'], df_final['Voltage (V)'], color='red', label='Extracted Data')
        plt.title(f"Extracted Correlation: Voltage vs Capacity\n(Scale: X={X_AXIS_MAX_VALUE}, Y={Y_AXIS_MAX_VALUE})")
        plt.xlabel("Capacity (mAh)")
        plt.ylabel("Voltage (V)")
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    digitizer = GraphDigitizer(IMAGE_PATH)
    digitizer.process()
    