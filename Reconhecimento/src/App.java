import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;

public class App {
    public static void main(String[] args) throws Exception {
        FaceRecognizer r = EigenFaceRecognizer.create();
    }

}
