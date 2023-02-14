for lr in 3e-4 3e-3 3e-2 3e-1 3e-5
do 
    for dropout in 0 0.3 0.4 0.5 0.6 0.7
    do
        for image_features in 25 50 100
        do
            python grid_search_models.py $lr $dropout $image_features 0,1 || exit 1

        done
    done
done