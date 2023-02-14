experiment_dict = {

    # tuned models
    'A_classifier_182p_tuned' : {
        'target': 'A',
        'data_dim': [182, 182, 218],
        'learning_rate': 3e-4,
        'dropout': 0.5,
        'image_features': 100,
        'epoch_count': 50
    },
    'T_classifier_182p_tuned' : {
        'target': 'T',
        'data_dim': [182, 182, 218],
        'learning_rate': 3e-4,
        'dropout': 0.5,
        'image_features': 100,
        'epoch_count': 50
    },
    'N_classifier_182p_tuned' : {
        'target': 'N',
        'data_dim': [182, 182, 218],
        'learning_rate': 3e-4,
        'dropout': 0.5,
        'image_features': 100,
        'epoch_count': 50
    },

    # Tuned w/ less epcohs
    'A_classifier_182p_tuned_less_epochs' : {
        'target': 'A',
        'data_dim': [182, 182, 218],
        'learning_rate': 3e-4,
        'dropout': 0.5,
        'image_features': 100,
        'epoch_count': 20
    },
    'T_classifier_182p_tuned_less_epochs' : {
        'target': 'T',
        'data_dim': [182, 182, 218],
        'learning_rate': 3e-4,
        'dropout': 0.5,
        'image_features': 100,
        'epoch_count': 20
    },
    'N_classifier_182p_tuned_less_epochs' : {
        'target': 'N',
        'data_dim': [182, 182, 218],
        'learning_rate': 3e-4,
        'dropout': 0.5,
        'image_features': 100,
        'epoch_count': 20
    },



    'A_classifier_182p_test' : {
        'target': 'A',
        'data_dim': [182, 182, 218],
        'epoch_count': 1
    },
    'A_classifier_182p_baseline' : {
        'target': 'A',
        'data_dim': [182, 182, 218]
    },

    'T_classifier_182p_baseline' : {
        'target': 'T',
        'data_dim': [182, 182, 218]
    },

    'N_classifier_182p_baseline' : {
        'target': 'N',
        'data_dim': [182, 182, 218]
    },

    'N_classifier_182p_batch_down' : {
        'target': 'N',
        'data_dim': [182, 182, 218],
        'batch_size': 8
    },
    'N_classifier_182p_batch_up' : {
        'target': 'N',
        'data_dim': [182, 182, 218],
        'batch_size': 32
    },
    'N_classifier_182p_lr_down' : {
        'target': 'N',
        'data_dim': [182, 182, 218],
        'learning_rate': 1e-5
    },
    'N_classifier_182p_lr_up' : {
        'target': 'N',
        'data_dim': [182, 182, 218],
        'learning_rate': 1e-3
    },
    'N_classifier_182p_decay_down' : {
        'target': 'N',
        'data_dim': [182, 182, 218],
        'decay': 10000
    },
    'N_classifier_182p_decay_up' : {
        'target': 'N',
        'data_dim': [182, 182, 218],
        'decay': 1000000
    },

}