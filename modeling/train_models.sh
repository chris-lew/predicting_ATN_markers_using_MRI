# python train_img_only_models.py A_classifier_182p_baseline 0,1 &&\
# python train_img_only_models.py T_classifier_182p_baseline 0,1 &&\
# python train_img_only_models.py N_classifier_182p_baseline 0,1 &&\

# python train_gaussian_labeled_img_only_models.py A_classifier_182p_baseline 2,3  &&\
# python train_gaussian_labeled_img_only_models.py T_classifier_182p_baseline 2,3 &&\
# python train_gaussian_labeled_models.py A_classifier_182p_tuned 2,3 --param True &&\
# python train_gaussian_labeled_img_only_models.py A_classifier_182p_tuned 2,3 --param True &&\

# python train_covariate_models.py N_classifier_182p_batch_down 0,1 &&\
# python train_covariate_models.py N_classifier_182p_batch_up 0,1,2 &&\
# python train_covariate_models.py N_classifier_182p_lr_down 0,1 &&\
# python train_covariate_models.py N_classifier_182p_lr_up 0,1 &&\
# python train_covariate_models.py N_classifier_182p_decay_down 0,1 &&\
# python train_covariate_models.py N_classifier_182p_decay_up 0,1 &&\

# python train_gaussian_labeled_models.py A_classifier_182p_tuned_less_epochs 1,3 --param True &&\
# python train_gaussian_labeled_img_only_models.py A_classifier_182p_tuned_less_epochs 1,3 --param True &&\

# python train_gaussian_labeled_models.py N_classifier_182p_tuned_less_epochs 1,3 --param True &&\
# python train_gaussian_labeled_img_only_models.py N_classifier_182p_tuned_less_epochs 1,3 --param True &&\

# python train_gaussian_labeled_models.py T_classifier_182p_tuned_less_epochs 1,3 --param True &&\
# python train_gaussian_labeled_img_only_models.py T_classifier_182p_tuned_less_epochs 1,3 --param True &&\

python train_gaussian_labeled_img_only_models.py N_classifier_182p_tuned 1,3 --param True &&


echo 'done'
