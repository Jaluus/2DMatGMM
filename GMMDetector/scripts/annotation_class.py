import os

import cv2
import numpy as np


class watershed_annotator:
    def __init__(self, image_directory, mask_directory):
        self.marks_updated = False

        self.l_mouse_down = False
        self.r_mouse_down = False

        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.image_names = [
            name
            for name in os.listdir(self.image_directory)
            if (name.endswith(".png") or name.endswith(".jpg"))
        ]
        self.image_paths = [
            os.path.join(self.image_directory, name) for name in self.image_names
        ]

        assert os.path.exists(
            self.image_directory
        ), "Image directory does not exist or the path is incorrect."
        assert os.path.exists(
            self.mask_directory
        ), "Mask directory does not exist or the path is incorrect."

    def run(self, current_idx=0):
        self.current_idx = current_idx

        self.current_image = cv2.imread(self.image_paths[current_idx])
        self.current_image_marked = np.copy(self.current_image)
        self.current_image_display = np.copy(self.current_image)
        self.marker_image = np.zeros(self.current_image.shape[:2], dtype=np.int32)
        self.watershed_segments = np.zeros(self.current_image.shape, dtype=np.uint8)

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.l_mouse_down = True

            if event == cv2.EVENT_LBUTTONUP:
                self.l_mouse_down = False

            if event == cv2.EVENT_RBUTTONDOWN:
                self.r_mouse_down = True

            if event == cv2.EVENT_RBUTTONUP:
                self.r_mouse_down = False

            if self.l_mouse_down:
                cv2.circle(self.marker_image, (x, y), 3, 1, -1)
                cv2.circle(self.current_image_marked, (x, y), 3, (0, 255, 0), -1)
                self.marks_updated = True

            if self.r_mouse_down:
                cv2.circle(self.marker_image, (x, y), 3, 2, -1)
                cv2.circle(self.current_image_marked, (x, y), 3, (0, 0, 255), -1)
                self.marks_updated = True

        def clear_marks():
            self.current_image_display = np.copy(self.current_image)
            self.current_image_marked = np.copy(self.current_image)
            self.marker_image = np.zeros(self.current_image.shape[0:2], dtype=np.int32)
            self.watershed_segments = np.zeros(self.current_image.shape, dtype=np.uint8)

        def update_current_image():
            self.current_image = cv2.imread(self.image_paths[current_idx])
            clear_marks()

        cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotator", mouse_callback)

        cv2.setWindowTitle("Annotator", self.image_paths[current_idx])

        while True:
            # Check if a mask already exists for the current image
            mask_exists = os.path.exists(
                os.path.join(self.mask_directory, self.image_names[current_idx])
            )
            window_title = f"{self.image_names[current_idx]} | {current_idx + 1 }/{len(self.image_paths)}"

            if mask_exists:
                window_title += " | Mask exists"

            cv2.imshow("Annotator", self.current_image_display)
            cv2.setWindowTitle(
                "Annotator",
                window_title,
            )

            key = cv2.waitKey(10)

            if key == 27:
                break

            elif key == ord("c"):
                clear_marks()

            if key == ord("s"):
                mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
                mask[marker_image_copy == 1] = 255
                cv2.imwrite(
                    os.path.join(self.mask_directory, self.image_names[current_idx]),
                    mask,
                )

            if key == ord("d"):
                if current_idx < len(self.image_paths) - 1:
                    current_idx += 1
                    update_current_image()

            if key == ord("a"):
                if current_idx > 0:
                    current_idx -= 1
                    update_current_image()

            # If we clicked somewhere, call the watershed algorithm on our chosen markers
            if self.marks_updated:
                self.current_image_display = self.current_image_marked.copy()
                marker_image_copy = self.marker_image.copy()

                # run the watershed algorithm with the chosen markers
                cv2.watershed(self.current_image, marker_image_copy)

                # create a mask of the watershed segments
                self.watershed_segments = np.zeros(
                    self.current_image.shape, dtype=np.uint8
                )
                self.watershed_segments[marker_image_copy == 1] = [0, 0, 255]
                self.watershed_segments[marker_image_copy == 2] = 0

                self.watershed_segments = cv2.morphologyEx(
                    self.watershed_segments,
                    cv2.MORPH_GRADIENT,
                    np.ones((3, 3), dtype=np.uint8),
                )

                self.current_image_display = cv2.addWeighted(
                    self.current_image_display, 1, self.watershed_segments, 1, 0
                )

                self.marks_updated = False

        cv2.destroyAllWindows()
