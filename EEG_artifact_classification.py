import numpy as np
import pandas as pd
import tensorflow as tf
import mne
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
from collections import Counter
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from mne.decoding import Vectorizer
from joblib import Parallel, delayed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Constants
RANDOM_STATE = 42

#######################
# Data Preprocessing  #
#######################

def load_and_stack_data(files):
    """Load and preprocess EEG data from multiple files."""
    stacked_artifacts = np.empty((0, 2))
    stacked_data = np.empty((0, 32, 2000))
    
    for fname in files:
        print(f"File: {fname}")
        
        # Load artifacts and data
        artifacts = ... # Load your labels
        data = ... # Load your data
        
        # Filter data
        data.filter(0.5, 20, method='iir')
        
        # Stack data
        stacked_artifacts = np.vstack((stacked_artifacts, artifacts))
        stacked_data = np.vstack((stacked_data, data))
    
    # Create epochs and apply baseline correction
    # Create info according to EEG setup
    info = mne.create_info(ch_names=[...], 
                          sfreq=..., 
                          ch_types='eeg')
    stacked_epochs = mne.EpochsArray(stacked_data, info, tmin=-2)
    stacked_epochs.crop(tmin=-0.5, tmax=1.0)
    stacked_epochs.apply_baseline(baseline=(-0.15, 0))
    
    return stacked_epochs.get_data(), stacked_artifacts[:, 1]

#####################################
# Machine Learning Grid Search      #
#####################################

def perform_ml_grid_search(X_train, y_train):
    """Perform grid search for Random Forest and SVM classifiers."""
    
    # Random Forest Grid Search
    rf_param_grid = {
        'n_estimators': [400, 500, 600, 700, 800, 900, 1000],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [45, 50, 55, None],
        'min_samples_split': [2, 3, 4, 5, 6, 7],
        'min_samples_leaf': [1, 2, 4, 5, 10]
    }
    
    rf_grid_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        rf_param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    rf_grid_search.fit(X_train, y_train)
    print("Best parameters for Random Forest:", rf_grid_search.best_params_)
    
    # SVM Grid Search
    svm_param_grid = {
        'C': [10**i for i in range(-3, 5)],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    svm_grid_search = GridSearchCV(
        SVC(),
        svm_param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    svm_grid_search.fit(X_train, y_train)
    print("Best parameters for SVM:", svm_grid_search.best_params_)

#############################
# Deep Learning Grid Search #
#############################

def build_cnn_lstm_model(conv_layers=2, filters=64, kernel_size=3, 
                        lstm_layers=1, bidirectional=False, dropout_rate=0.5):
    """Build CNN-LSTM model with specified parameters."""
    model = Sequential()
    
    # Convolutional layers
    for _ in range(conv_layers):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, 
                        activation='relu', padding='same'))
    
    # LSTM layers
    for _ in range(lstm_layers):
        lstm_layer = LSTM(64, return_sequences=True)
        if bidirectional:
            model.add(Bidirectional(lstm_layer))
        else:
            model.add(lstm_layer)
    
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def perform_dl_grid_search(X, y):
    """Perform grid search for CNN-LSTM hyperparameters."""
    param_options = {
        'conv_layers': [2, 3, 4],
        'filters': [64, 128, 256, 512],
        'kernel_size': [1, 2, 3, 4, 5],
        'lstm_layers': [1, 2],
        'bidirectional': [False, True],
        'dropout_rate': [0.3, 0.5, 0.7]
    }
    
    for conv_layers in param_options['conv_layers'][:2]:
        for filters in param_options['filters'][:2]:
            for kernel_size in param_options['kernel_size'][:2]:
                for lstm_layers in param_options['lstm_layers']:
                    for bidirectional in param_options['bidirectional'][:1]:
                        for dropout_rate in param_options['dropout_rate'][:2]:
                            print(f"Testing: Conv={conv_layers}, Filters={filters}, "
                                  f"Kernel={kernel_size}, LSTM={lstm_layers}, "
                                  f"Bi={bidirectional}, Dropout={dropout_rate}")
                            
                            model = build_cnn_lstm_model(conv_layers, filters, kernel_size,
                                                       lstm_layers, bidirectional, dropout_rate)
                            model.fit(X, y, epochs=5, batch_size=32, 
                                    validation_split=0.2, 
                                    callbacks=[EarlyStopping(patience=2)], 
                                    verbose=1)

#######################
# Deep Learning Model #
#######################

def train_dl_model(X, y):
    """Train and evaluate CNN-LSTM model with cross-validation."""
    num_epochs, num_channels, num_time_points = X.shape
    
    # Normalize data
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, num_time_points)
    X_normalized = scaler.fit_transform(X_reshaped)
    X = X_normalized.reshape(num_epochs, num_channels, num_time_points)
    
    # Model parameters
    params = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'filters': 128,
        'kernel_size': 2,
        'model_variation': 'simple_cnn_lstm'
    }
    
    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f'Starting fold {fold}')
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
        
        # Build model
        model = Sequential([
            Conv1D(filters=params['filters'], kernel_size=params['kernel_size'],
                  activation='relu', input_shape=(num_channels, num_time_points)),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            Conv1D(filters=params['filters']*2, kernel_size=params['kernel_size'],
                  activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            LSTM(64),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile and train
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                     loss='binary_crossentropy', metrics=['accuracy'])
        
        start_time = time.time()
        history = model.fit(X_train, y_train, 
                          epochs=20, 
                          batch_size=params['batch_size'],
                          validation_data=(X_val, y_val),
                          callbacks=[EarlyStopping(monitor='val_loss', 
                                                 patience=10, 
                                                 restore_best_weights=True)],
                          verbose=1)
        
        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        
        results.append({
            'fold': fold,
            **params,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'time_taken_seconds': time.time() - start_time
        })
        print(f'Completed fold {fold}')
    
    return pd.DataFrame(results)

#################
# Random Forest #
#################

def rf_train_test(train, test, X, y, clf):
    """Train and test Random Forest classifier for one CV split."""
    clf.fit(X[train], y[train])
    preds = clf.predict(X[test])
    
    tn, fp, fn, tp = confusion_matrix(y[test], preds).ravel()
    indices = {
        'tp': test[(preds == 1) & (y[test] == 1)],
        'tn': test[(preds == 0) & (y[test] == 0)],
        'fp': test[(preds == 1) & (y[test] == 0)],
        'fn': test[(preds == 0) & (y[test] == 1)]
    }
    
    return (accuracy_score(y[test], preds),
            balanced_accuracy_score(y[test], preds),
            f1_score(y[test], preds),
            f1_score(y[test], preds, pos_label=0),
            tn, fp, fn, tp,
            *indices.values())

def apply_random_forest(X, y):
    """Apply Random Forest classification with cross-validation."""
    clf = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        RandomForestClassifier(
            bootstrap=False,
            max_depth=None,
            max_features='sqrt',
            min_samples_leaf=2,
            min_samples_split=7,
            n_estimators=800,
            n_jobs=50,
            random_state=RANDOM_STATE
        )
    )
    
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_STATE)
    results = Parallel(n_jobs=50)(
        delayed(rf_train_test)(train, test, X, y, clf) 
        for train, test in cv.split(X, y)
    )
    
    metrics = [np.mean(col) for col in zip(*results)][:8]
    indices = [np.concatenate(col) for col in zip(*results)][8:]
    
    return (*metrics, *indices)

##########################
# Support Vector Machine #
##########################

def svm_train_test(train, test, X, y, clf):
    """Train and test SVM classifier"""
    clf.fit(X[train], y[train])
    preds = clf.predict(X[test])
    
    tn, fp, fn, tp = confusion_matrix(y[test], preds).ravel()
    results_with_proba = ['TP' if (t == p == 1) else 
                         'TN' if (t == p == 0) else 
                         'FP' if (t == 0 and p == 1) else 
                         'FN' 
                         for t, p in zip(y[test], preds)]
    
    return (accuracy_score(y[test], preds),
            balanced_accuracy_score(y[test], preds),
            f1_score(y[test], preds),
            f1_score(y[test], preds, pos_label=0),
            tn, fp, fn, tp,
            results_with_proba)

def apply_svm(X, y):
    """Apply SVM classification with cross-validation."""
    clf = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1000,
            gamma='scale',
            probability=False,
            verbose=True
        )
    )
    
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_STATE)
    results = Parallel(n_jobs=50)(
        delayed(svm_train_test)(train, test, X, y, clf) 
        for train, test in cv.split(X, y)
    )
    
    metrics = [np.mean(col) for col in zip(*results)][:8]
    proba_results = [item for sublist in [r[8] for r in results] for item in sublist]
    
    return (*metrics, proba_results)


###############################
# Training Size Analysis - DL #
###############################

def train_dl_model_with_sizes(X, y, training_sizes):
    """Train and evaluate CNN-LSTM model across different training sizes."""
    num_epochs, num_channels, num_time_points = X.shape
    
    # Normalize data (from original train_dl_model)
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, num_time_points)
    X_normalized = scaler.fit_transform(X_reshaped)
    X = X_normalized.reshape(num_epochs, num_channels, num_time_points)
    
    # Model parameters (from original train_dl_model)
    params = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'filters': 128,
        'kernel_size': 2,
        'model_variation': 'simple_cnn_lstm'
    }
    
    results = []
    
    # Iterate over training sizes (from new snippet)
    for train_size in training_sizes:
        print(f'Starting analysis for training size: {train_size}')
        
        # Split the data into training and remaining (validation + test) sets
        X_train, X_rem, y_train, y_rem = train_test_split(
            X, y, train_size=train_size, random_state=RANDOM_STATE, stratify=y)
        
        # Further split the remaining data into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(
            X_rem, y_rem, test_size=0.5, random_state=RANDOM_STATE, stratify=y_rem)
        
        start_time = time.time()
        
        # Build model (exact architecture from original train_dl_model)
        model = Sequential([
            Conv1D(filters=params['filters'], kernel_size=params['kernel_size'],
                   activation='relu', input_shape=(num_channels, num_time_points)),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            Conv1D(filters=params['filters']*2, kernel_size=params['kernel_size'],
                   activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.5),
            LSTM(64),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model (from original train_dl_model)
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                     loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train model (adapted from original train_dl_model, but with single split)
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=1
        )
        
        # Evaluate model (from original train_dl_model)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
        
        # Calculate metrics
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        duration = time.time() - start_time
        
        # Store results
        results.append({
            'training_size': train_size,
            **params,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'time_taken_seconds': duration
        })
        print(f'Completed training size {train_size}: balanced_accuracy={balanced_acc:.4f}, '
              f'f1_score={f1:.4f}, time_taken={duration:.2f} seconds')
    
    return pd.DataFrame(results)


###############################
# Training Size Analysis - RF #
###############################


def train_and_test(X_train, X_test, y_train, y_test, clf):
    """Train and evaluate the classifier on given train/test split."""
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return balanced_accuracy_score(y_test, preds)

def apply_random_forest_trainsize(X, y, training_sizes):
    """Apply Random Forest classification across different training sizes."""

    clf = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        RandomForestClassifier(
            bootstrap=False,
            max_depth=None,
            max_features='sqrt',
            min_samples_leaf=2,
            min_samples_split=7,
            n_estimators=800,
            n_jobs=50,
            random_state=RANDOM_STATE
        )
    )
    
    results = []
    
    # Iterate over training sizes
    for train_size in training_sizes:
        print(f'Starting analysis for training size: {train_size}')
        
        start_time = time.time()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=RANDOM_STATE, stratify=y)
        
        # Train and evaluate
        balanced_acc = train_and_test(X_train, X_test, y_train, y_test, clf)
        
        duration = time.time() - start_time
        
        results.append({
            'training_size': train_size,
            'balanced_accuracy': balanced_acc,
            'time_taken_seconds': duration
        })
        print(f'Completed training size {train_size}: '
              f'balanced_accuracy={balanced_acc:.4f}, time_taken={duration:.2f} seconds')
    
    results_df = pd.DataFrame(results)
    return results_df[['training_size', 'balanced_accuracy']]


###########################
# Plot train size effects #
###########################

LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14
FIGURE_SIZE = (10, 6)
COLOR_DL = '#ff7f0e'  # Orange for Deep Learning
COLOR_RF = '#1f77b4'  # Blue for Random Forest

def plot_training_size_analysis(deep_learning_results, random_forest_results):
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(
        deep_learning_results['training_size'],
        deep_learning_results['balanced_accuracy'],
        marker='o',
        color=COLOR_DL,
        label='Deep Learning Classifier',
        linestyle='-',
        linewidth=2
    )
    
    plt.plot(
        random_forest_results['training_size'],
        random_forest_results['balanced_accuracy'],
        marker='o',
        color=COLOR_RF,
        label='Random Forest Classifier',
        linestyle='-',
        linewidth=2
    )
    
    plt.xscale('log')
    plt.xticks(
        deep_learning_results['training_size'],
        labels=deep_learning_results['training_size'],
        rotation=45,
        fontsize=TICK_FONTSIZE
    )
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())
    plt.yticks(fontsize=TICK_FONTSIZE)
    
    plt.xlabel('Training Size (log)', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Balanced Accuracy', fontsize=LABEL_FONTSIZE)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    
    plt.legend(fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout()
    plt.savefig('Training_Size_Analysis_v01_600dpi.png', dpi=600)
    
    plt.show()


###############################################
# Certainty / classifier probability analysis #
###############################################

# Run previous RF and DL classification and save probability scores for this anaylsis
deeplearning_df = ... # df with results including probability scores

# convert probabilities to make them comparable to random forest probas.
# before: probabilities range between 0 and 1, which values closer to 0 being high indication for class 0, and >.5 for class 1
# now: all probas are between 0.5 (chance) and 1 (certain), independent of class
deeplearning_df['Probabilities'] = deeplearning_df['Probabilities'].apply(lambda item: 1 - item if item < 0.5 else item)

# Calculate classification performance changes at different confidence thresholds

# Define a range of possible cutoff values from 0.5 to 1.0
cutoffs = np.linspace(0.5, 1.0, num=51)
metrics_dl = []

for cutoff in cutoffs:
    # Filter the dataframe to include only samples where the probability is above the cutoff
    df_filtered = deeplearning_df[deeplearning_df['Probabilities'] >= cutoff]
    
    # If no samples are left, skip this cutoff
    if df_filtered.empty:
        continue
    
    # Calculate confusion matrix
    cm = confusion_matrix(df_filtered['TrueLabels'], df_filtered['PredictedLabels'])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(cm) == 1:
            if df_filtered['TrueLabels'].iloc[0] == 1:
                tp = cm[0][0]
            else:
                tn = cm[0][0]

    # Calculate the percentage of the sample included
    sample_percentage = (deeplearning_df['Probabilities'] >= cutoff).mean() * 100
    
    # Calculate accuracy
    accuracy = accuracy_score(df_filtered['TrueLabels'], df_filtered['PredictedLabels'])
    
    # Calculate the percentage of samples above the cutoff that belong to class 1
    percentage_class_1 = (df_filtered['TrueLabels'] == 1).mean() * 100
    
    metrics_dl.append((cutoff, sample_percentage, accuracy, percentage_class_1))

# Convert to DataFrame
metrics_deeplearning_df = pd.DataFrame(metrics_dl, columns=['cutoff', 'sample_percentage', 'accuracy', 'percentage_class_1'])

# Random Forest:
# all_results_with_proba are the results of RF with probabilities enabled

df = pd.DataFrame(all_results_with_proba, columns=['result_type', 'true_label', 'pred_label', 'probability'])

df['max_probability'] = df['probability']

# Define a range of possible cutoff values from 0.5 to 1.0
cutoffs = np.linspace(0.5, 1.0, num=51)
metrics = []

total_samples = len(df)

for cutoff in cutoffs:
    # Filter the dataframe to include only samples where the probability is above the cutoff
    df_filtered = df[df['max_probability'] >= cutoff]
    
    # If no samples are left, skip this cutoff
    if df_filtered.empty:
        continue
    
    # Calculate confusion matrix
    cm = confusion_matrix(df_filtered['true_label'], df_filtered['pred_label'])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(cm) == 1:
            if df_filtered['true_label'].iloc[0] == 1:
                tp = cm[0][0]
            else:
                tn = cm[0][0]

    # Calculate the percentage of the sample included
    sample_percentage = (df['max_probability'] >= cutoff).mean() * 100
    
    # Calculate accuracy
    accuracy = accuracy_score(df_filtered['true_label'], df_filtered['pred_label'])
    
    # Calculate the percentage of class 1 values in true labels
    class_1_percentage = (df_filtered['true_label'] == 1).mean() * 100
    
    # Calculate the percentages for TP, FP, TN, FN
    total_filtered = len(df_filtered)
    tp_pct = (tp / total_filtered) * 100
    fp_pct = (fp / total_filtered) * 100
    tn_pct = (tn / total_filtered) * 100
    fn_pct = (fn / total_filtered) * 100
    
    metrics.append((cutoff, tp_pct, fp_pct, tn_pct, fn_pct, sample_percentage, accuracy, class_1_percentage))

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(metrics, columns=['cutoff', 'TP_pct', 'FP_pct', 'TN_pct', 'FN_pct', 'sample_percentage', 'accuracy', 'class_1_percentage'])


##################
# Plot certainty #
##################

label_fontsize = 16
tick_fontsize = 14
legend_fontsize = 14

# Plot the metrics with a secondary y-axis for accuracy
fig, ax1 = plt.subplots(figsize=(12, 8))

# Define colors
colors_rf = {
    'Sample': '#1f77b4',
    'Accuracy': '#1f77b4'
}

colors_dl = {
    'Sample': '#ff7f0e',
    'Accuracy': '#ff7f0e'
}

# Plot sample percentage for both classifiers
ax1.plot(metrics_df['cutoff'], metrics_df['sample_percentage'], label='RF sample above threshold (%)',color=colors_rf['Sample'], linewidth=2)
ax1.plot(metrics_deeplearning_df['cutoff'], metrics_deeplearning_df['sample_percentage'], label='DL sample above threshold (%)', color=colors_dl['Sample'], linewidth=2)

# Create a secondary y-axis for accuracy
ax2 = ax1.twinx()

# Plot accuracy for both classifiers
ax2.plot(metrics_df['cutoff'], metrics_df['accuracy'], label='RF accuracy', linestyle='-', color=colors_rf['Accuracy'], linewidth=2, marker='o')
ax2.plot(metrics_deeplearning_df['cutoff'], metrics_deeplearning_df['accuracy'], label='DL accuracy', linestyle='-', color=colors_dl['Accuracy'], linewidth=2, marker='o')

# Set labels with adjustable font size
ax2.set_ylabel('Accuracy', fontsize=label_fontsize)
ax1.set_xlabel('Probability score', fontsize=label_fontsize)
ax1.set_ylabel('Percentage', fontsize=label_fontsize)

# Increase the font size of ticks
ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)

# Adjust the left y-axis to match the right y-axis scale
ax1.set_ylim(0, 100)  # Scale from 0 to 100 for percentages
ax1.set_yticks(np.linspace(0, 100, 11))  # Set ticks from 0 to 100

ax1.grid(True)

# Combine legends and adjust order
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Manually reorder legends to have Random Forest lines first
lines = [lines2[0], lines1[0], lines2[1], lines1[1]]
labels = [labels2[0], labels1[0], labels2[1], labels1[1]]

ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.55, 0), fontsize=legend_fontsize)

plt.show()    





# Example usage
if __name__ == "__main__":
    
    files = []  # List your files
    X, y = load_and_stack_data(files)
    
    perform_ml_grid_search(X, y)
    perform_dl_grid_search(X, y)
    dl_results = train_dl_model(X, y)
    rf_results = apply_random_forest(X, y)
    svm_results = apply_svm(X, y)

    # training size
    training_sizes = [10, 20, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
    dl_trainsize_results = train_dl_model_with_sizes(X, y, training_sizes)
    rf_trainsize_results = apply_random_forest_trainsize(X, y, training_sizes)

    # plot training size
    plot_training_size_analysis(dl_results, rf_results)




