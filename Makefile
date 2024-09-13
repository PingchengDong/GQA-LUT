gelu_8:
	python gqa_lut_trainer.py --act_func 'gelu' --x_range -4 4 --sp_range -4.0 4.0 --num_splits 7 --decimal_bit_range 0 6 --total_iters 500 --mutate

gelu_16:
	python gqa_lut_trainer.py --act_func 'gelu' --x_range -4 4 --sp_range -4.0 4.0 --num_splits 15 --decimal_bit_range 0 6 --total_iters 500 --mutate

hswish_8:
	python gqa_lut_trainer.py --act_func 'hswish' --x_range -4 4 --sp_range -4.0 4.0 --num_splits 7 --decimal_bit_range 0 6 --total_iters 500 --mutate

hswish_16:
	python gqa_lut_trainer.py --act_func 'hswish' --x_range -4 4 --sp_range -4.0 4.0 --num_splits 15 --decimal_bit_range 0 6 --total_iters 500 --mutate

exp_8:
	python gqa_lut_trainer.py --act_func 'exp' --x_range -8.0 0 --sp_range -8.0 0.0 --num_splits 7 --decimal_bit_range 0 6 --total_iters 500 --mutate

exp_16:
	python gqa_lut_trainer.py --act_func 'exp' --x_range -8.0 0 --sp_range -6.5 0.0 --num_splits 15 --decimal_bit_range 0 6 --total_iters 500 --mutate

reci_8:
	python gqa_lut_trainer.py --act_func 'reci' --x_range 0.5 4.0 --sp_range 0.625 3.875 --num_splits 7 --decimal_bit_range 5 5 --total_iters 500

reci_16:
	python gqa_lut_trainer.py --act_func 'reci' --x_range 0.5 4.0 --sp_range 0.625 3.875 --num_splits 15 --decimal_bit_range 5 5 --total_iters 500

sqrt_reci_8:
	python gqa_lut_trainer.py --act_func 'sqrt_reci' --x_range 0.25 4.0 --sp_range 0.25 3.875 --num_splits 7 --decimal_bit_range 0 5 --total_iters 500

sqrt_reci_16:
	python gqa_lut_trainer.py --act_func 'sqrt_reci' --x_range 0.25 4.0 --sp_range 0.25 3.875 --num_splits 15 --decimal_bit_range 0 5 --total_iters 500