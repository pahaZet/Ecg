import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from keras.models import load_model
from keras.optimizers import Adam
from datasets_original import ECGSequence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('--path_to_hdf5', type=str, default='C:\\paha\ECG\\Code15data\\CheckupDatasets\\ecg_tracings.hdf5',
                        help='path to hdf5 file containing tracings')
    parser.add_argument('--path_to_model', type=str, default='C:\\paha\\ECG\\Code15data\\model.hdf5',  # or model_date_order.hdf5
                        help='file containing training model.')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--output_file', default="./dnn_output.npy",  # or predictions_date_order.csv
                        help='output csv file.')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size.')

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    # Import data
    seq = ECGSequence(args.path_to_hdf5, args.dataset_name, batch_size=args.bs)
    # Import model
    model = load_model(args.path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    y_score = model.predict(seq,  verbose=1)


    for idx, yii in enumerate(y_score):
        a = np.asarray(yii)
        if a.max() > 0.5:        
            print("idx ", idx, " max ", a.max())
            print(a)
            print("")
            

    # Generate dataframe
    np.save(args.output_file, y_score)

    print("Output predictions saved")