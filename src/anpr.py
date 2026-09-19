import cv2
import re
import numpy as np

from paddleocr import PaddleOCR


class ANPREngine:

    VEHICLE_CLASSES = {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "motorbike"
    }

    def __init__(
        self,
        confidence_threshold=0.60,
        device="cpu"
    ):
        self.confidence_threshold = confidence_threshold

        # PaddleOCR 3.x
        #
        # IMPORTANT:
        # Do NOT use use_gpu=...
        self.ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            device="cpu"

        )

    @staticmethod
    def _preprocess(vehicle_roi):

        if vehicle_roi is None or vehicle_roi.size == 0:
            return None

        # Resize
        scale = 2.0

        resized = cv2.resize(
            vehicle_roi,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        # Slight enhancement
        gray = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        enhanced = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

        return enhanced

    @staticmethod
    def _clean_plate_text(text):

        if not text:
            return None

        text = str(text).upper()

        # Remove spaces and special characters.
        text = re.sub(
            r"[^A-Z0-9]",
            "",
            text
        )

        if len(text) < 4:
            return None

        return text

    def read_plate(self, vehicle_roi):

        processed = self._preprocess(vehicle_roi)

        if processed is None:
            return None, 0.0

        try:
            results = self.ocr.predict(processed)

            best_text = None
            best_score = 0.0

            for result in results:

                # PaddleOCR 3.x result object.
                data = None

                try:
                    data = result.json
                except Exception:
                    pass

                if callable(data):
                    data = data()

                # Some versions return a dictionary.
                if isinstance(data, dict):

                    # Most common PaddleOCR 3.x structure:
                    # {"res": {...}}
                    if "res" in data:
                        data = data["res"]

                    texts = data.get(
                        "rec_texts",
                        []
                    )

                    scores = data.get(
                        "rec_scores",
                        []
                    )

                    for text, score in zip(
                        texts,
                        scores
                    ):

                        try:
                            score = float(score)
                        except Exception:
                            continue

                        clean_text = self._clean_plate_text(
                            text
                        )

                        if (
                            clean_text
                            and score >= self.confidence_threshold
                            and score > best_score
                        ):
                            best_text = clean_text
                            best_score = score

                # Fallback for result objects that expose
                # attributes directly.
                else:

                    try:
                        texts = result.rec_texts
                        scores = result.rec_scores

                        for text, score in zip(
                            texts,
                            scores
                        ):

                            score = float(score)

                            clean_text = self._clean_plate_text(
                                text
                            )

                            if (
                                clean_text
                                and score >= self.confidence_threshold
                                and score > best_score
                            ):
                                best_text = clean_text
                                best_score = score

                    except Exception:
                        continue

            return best_text, best_score

        except Exception as exc:

            print(
                f"[ANPR] OCR error: {exc}"
            )

            return None, 0.0