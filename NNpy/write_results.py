"""
    Module to create output project file and train final model with blind data
"""
import csv
from input_reading import read_cup, read_test_cup

train_data, train_labels, _, _ = read_cup(frac_train=0.8)
test_index, test_data = read_test_cup()


def create_output_file(file_path, test_prediction):
    with open(file_path, 'w', newline="") as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['# Roberto Esposito, Marco Mazzei, Antonio Zegarelli'])
        writer.writerow(['# Team_Overflow'])
        writer.writerow(['# ML-CUP21'])
        writer.writerow(['# 03/01/2021'])

        for index, elem in zip(test_index, test_prediction):
            writer.writerow([
                str(index),
                elem[0],
                elem[1],
            ])


"""
create_output_file("Team_Overflow_ML-CUP21-TS.csv",
                   ensemble.predict(test_data))
"""