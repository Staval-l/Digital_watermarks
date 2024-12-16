from itertools import count

import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
import multiprocessing
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score


def read_img(path_to_img) -> np.ndarray:
    return cv2.imread(path_to_img, cv2.IMREAD_GRAYSCALE)


def lsb_embled(img: np.ndarray, q: float) -> np.ndarray:
    res = img.copy()
    num_pixels_to_change = round(img.shape[0] * img.shape[1] * q)
    noise = np.random.uniform(0, 1, num_pixels_to_change).round().astype(np.uint8)
    pixels_to_change = np.random.randint(0, img.shape[0] * img.shape[1], num_pixels_to_change)
    bit_img = np.unpackbits(res, bitorder='little').reshape((512, 512, 8))
    for i in range(pixels_to_change.shape[0]):
        needed_ind = divmod(pixels_to_change[i], img.shape[0])
        bit_img[needed_ind[1], needed_ind[0], 2] = noise[i]
    res = np.packbits(bit_img.reshape((512, 512 * 8)), bitorder='little').reshape((512, 512))
    return res


def ds_preparation(path_to_dir: str, q_list: list, output_folder: str) -> None:
    for q in tqdm(q_list):
        output_dir_path = output_folder + str(q)
        os.makedirs(output_dir_path, exist_ok=True)
        num_images_to_process = len(os.listdir(path_to_dir)) // 2
        for filename in os.listdir(path_to_dir)[:num_images_to_process]:
            image_path = os.path.join(path_to_dir, filename)
            new_img = lsb_embled(read_img(image_path), q)
            output_path = os.path.join(output_dir_path, filename)
            cv2.imwrite(output_path, new_img)
        for filename in os.listdir(path_to_dir)[num_images_to_process:]:
            image_path = os.path.join(path_to_dir, filename)
            shutil.copy(image_path, output_dir_path)


def serpentine_sweep(img: np.ndarray) -> np.ndarray:
    res = img[0][:]
    for i in range(1, img.shape[0]):
        if i % 2 == 0:
            res = np.concatenate((res, img[i][0:512]), axis=0)
        if i % 2 == 1:
            res = np.concatenate((res, img[i][:512][::-1]), axis=0)
    return res


def vector_init(img: np.ndarray) -> dict:
    bit_img = np.unpackbits(img, bitorder='little').reshape((512, 512, 8))
    sweep = serpentine_sweep(bit_img[:, :, 2])
    counts = []
    count = 1
    for i in range(sweep.shape[0]):
        if sweep[i - 1] == sweep[i]:
            count += 1
        else:
            counts.append(count)
            count = 1
    counts.append(count)
    res_dict = Counter(counts)
    res_dict = dict(sorted(res_dict.items()))
    return res_dict


def preparing_features(path_to_dir: str):
    features = []
    labels = []
    max_ser = []
    for filename in tqdm(os.listdir(path_to_dir)):
        image_path = os.path.join(path_to_dir, filename)
        img = read_img(image_path)
        res_dict = vector_init(img)
        keys = [int(key) for key in res_dict.keys()]
        max_ser.append(max(keys))
        max_run_length = 55291  # получено опытным путём 55291 на всех q
        feature_vector = [res_dict.get(i, 0) for i in range(1, max_run_length + 1)]
        features.append(feature_vector)
        file_number = int(filename[5:10])
        labels.append(0) if file_number > 500 else labels.append(1)
    return np.array(features), np.array(labels)


def tune_svc_hyperparameters(x_train, y_train, x_test, y_test, clf):
    param_grid = {
        'C': [0.1, 1, 10, 100]
    }
    n_jobs = multiprocessing.cpu_count() - 3
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid,
                               scoring='accuracy', cv=5, n_jobs=n_jobs)
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Лучшие параметры:", best_params)
    for params in grid_search.cv_results_['params']:
        model = SVC(**params, probability=True)
        model.fit(x_train, y_train)
        y_scores = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2,
                 label='ROC кривая (AUC = %0.2f) для параметров %s' % (roc_auc, params))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Доля ложноположительных результатов')
    plt.ylabel('Доля истинноположительных результатов')
    plt.title('ROC-кривые для различных комбинаций параметров SVC')
    plt.legend(loc="lower right")
    plt.show()

    return best_params, best_model


def train_model(path_to_dir: str, q_list: list) -> None:  # 84.67%
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for q in q_list:
        print(f"-----------------------------------------{q}-----------------------------------------")
        features, labels = preparing_features(path_to_dir + str(q))
        x_train = np.concatenate((features[:round(500 * 0.7)], features[500:500 + round(500 * 0.7)]))
        y_train = np.concatenate((labels[:round(500 * 0.7)], labels[500:500 + round(500 * 0.7)]))
        x_test = np.concatenate((features[500 + round(500 * 0.7): features.shape[0]], features[round(500 * 0.7):500]))
        y_test = np.concatenate((labels[500 + round(500 * 0.7): labels.shape[0]], labels[round(500 * 0.7):500]))

        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        clf = SVC(C= 100, kernel='linear', probability=True, random_state=52)

        clf.fit(x_train_scaled, y_train)
        # dump(clf, f'model_{q}.joblib')  # save clf, for load --> clf = load('filename.joblib')
        y_pred = clf.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f'Точность классификации: {accuracy:.2%}')
        print(f'Precision: {precision:.2%}')
        print(f'Recall: {recall:.2%}')
        print(f'F1 Score: {f1:.2%}')
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    plt.figure(figsize=(10, 6))
    plt.plot(q_list, accuracy_list, marker='o', label='Accuracy')
    plt.plot(q_list, precision_list, marker='o', label='Precision')
    plt.plot(q_list, recall_list, marker='o', label='Recall')
    plt.plot(q_list, f1_list, marker='o', label='F1 Score')
    plt.title('Performance Metrics vs. Parameter q')
    plt.xlabel('q')
    plt.ylabel('Score')
    plt.xticks(q_list)
    plt.grid()
    plt.legend()
    plt.show()


def dop(path_to_dir: str) -> None:
    features, labels = preparing_features(path_to_dir)
    x_train = np.concatenate((features[:round(500 * 0.7)], features[500:500 + round(500 * 0.7)]))
    y_train = np.concatenate((labels[:round(500 * 0.7)], labels[500:500 + round(500 * 0.7)]))
    x_test = np.concatenate((features[500 + round(500 * 0.7): features.shape[0]], features[round(500 * 0.7):500]))
    y_test = np.concatenate((labels[500 + round(500 * 0.7): labels.shape[0]], labels[round(500 * 0.7):500]))

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf = SVC(C=10, kernel='linear', probability=True, random_state=52)
    tune_svc_hyperparameters(x_train_scaled, y_train, x_test_scaled, y_test, clf)


def main():
    img = read_img(r'C:\Users\vodnyy\PycharmProjects\watermarking\lab4\ds_q_0.2\Image00001.tif')
    path_to_dir = 'BOWS2'
    output_folder = 'ds_q_'
    q_list = [0.2, 0.4, 0.6, 0.8, 1]
    train_model(output_folder, q_list)
    # dop(output_folder + str(q_list[0]))


if __name__ == '__main__':
    main()
