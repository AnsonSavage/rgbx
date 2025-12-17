cd /grphome/grp_cs_650_rgb_x

/grphome/grp_cs_650_rgb_x/conda_env/bin/python3 rgbx/x2rgb/batch_text_only_x2rgb.py \
  --checkpoint "" \
  --checkpoint "/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/checkpoints/x2rgb-finetuned/checkpoint-50" \
  --checkpoint "/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/checkpoints/x2rgb-finetuned/checkpoint-550" \
  --checkpoint "/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/checkpoints/x2rgb-finetuned_discrete_with_velocity/checkpoint-50" \
  --checkpoint "/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/checkpoints/x2rgb-finetuned_discrete_with_velocity/checkpoint-350" \
  --checkpoint "/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/checkpoints/x2rgb-finetuned_mixed_mixed_with_velocity_random_material/checkpoint-100/unet" \
  --checkpoint "/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/checkpoints/x2rgb-finetuned_mixed_mixed_with_velocity_random_material/checkpoint-10600/unet" \
  --checkpoint "/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/checkpoints/x2rgb-finetuned_mixed_mixed_with_velocity_standby_test/checkpoint-100/unet" \
  --checkpoint "/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/checkpoints/x2rgb-finetuned_mixed_mixed_with_velocity_standby_test/checkpoint-2300/unet" \
  --checkpoint "/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/checkpoints/x2rgb-finetuned_prompt_dropout/checkpoint-50" \
  --checkpoint "/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/checkpoints/x2rgb-finetuned_prompt_dropout/checkpoint-1250" \
  --aov_folder "/home/ansonsav/masters_thesis/contrastive_lighting_dataset_creation_utils/contrastive_data/discrete_renders/product_content_lock_test_varied_materials_05_this_would_be_a_good_validation_set/bcb89eac-d490-4271-a903-abc7f196559e/aovs/bcb89eac-d490-4271-a903-abc7f196559e_cam_28_scatter_49_objsel_70_mat_88_aovs" \
  --aov_folder "/grphome/grp_cs_650_rgb_x/rgbx/x2rgb/example/car_studio_512" \
  --aov_folder "/grphome/grp_cs_650_rgb_x/rgbx/x2rgb/example/kitchen_512" \
  --aov_folder "/grphome/grp_cs_650_rgb_x/rgbx/x2rgb/example/bedroom_512" \
  --prompt "soft directional lighting from the left" \
  --prompt "golden hour lighting" \
  --prompt "cool blue rim lighting" \
  --prompt "beautiful green lighting from the left" \
  --output_root "rgbx/x2rgb/outputs/compare_models_final_for_paper" \
  --run_name "compare_models_08" \
  --seed 123 \
  --num_samples 4
