import onnxruntime as ort


if __name__ == "__main__":
    onnx_path = '.\\weights\\yolop-640-640-v4.2.onnx'
    try:
        sess = ort.InferenceSession(onnx_path)

        for ii in sess.get_inputs():
            print("Input: ", ii)
        for oo in sess.get_outputs():
            print("Output: ", oo)

        print('read onnx using onnxruntime sucess')
    except Exception as e:
        print('read failed')
        raise e