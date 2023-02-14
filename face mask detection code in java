import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class FaceMaskDetection {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_default.xml");
        CascadeClassifier maskDetector = new CascadeClassifier("haarcascade_mask.xml");

        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("Error: Could not open camera");
            System.exit(1);
        }

        Mat frame = new Mat();
        while (true) {
            if (camera.read(frame)) {
                Mat grayFrame = new Mat();
                Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

                MatOfRect faceDetections = new MatOfRect();
                faceDetector.detectMultiScale(grayFrame, faceDetections);

                for (Rect rect : faceDetections.toArray()) {
                    Imgproc.rectangle(frame, new Point(rect.x, rect.y),
                            new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));

                    Mat faceROI = grayFrame.submat(rect);
                    MatOfRect maskDetections = new MatOfRect();
                    maskDetector.detectMultiScale(faceROI, maskDetections);

                    for (Rect mask : maskDetections.toArray()) {
                        Imgproc.rectangle(frame, new Point(rect.x + mask.x, rect.y + mask.y),
                                new Point(rect.x + mask.x + mask.width, rect.y + mask.y + mask.height),
                                new Scalar(0, 0, 255));
                    }
                }

                System.out.println(String.format("Face mask detection: %s faces found", faceDetections.toArray().length));

                Imgproc.putText(frame, "Press 'q' to quit", new Point(10, 25), Core.FONT_HERSHEY_SIMPLEX, 0.7,
                        new Scalar(255, 0, 0), 2);
                Imgproc.imshow("Face Mask Detection", frame);

                char key = (char) Imgproc.waitKey(25);
                if (key == 'q') {
                    break;
                }
            }
        }

        camera.release();
    }
}
