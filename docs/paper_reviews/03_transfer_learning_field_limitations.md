# Paper review 3: Transfer learning and field limitations

## Citation

Tm, P., Pranathi, A., SaiAshritha, K., Chittaragi, N. B., & Koolagudi, S. G. (2022). *Transfer learning for versatile plant disease recognition with limited data.* Plant Methods, 18, 24. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9726777/)

## Review

The paper examines transfer learning for plant disease recognition and discusses the gap between controlled datasets such as PlantVillage and field images. It motivates realistic augmentation and cautious interpretation of confidence scores.

## Project decision

Apply horizontal flips, small rotations, and modest colour jitter only to training images. Keep validation/test preprocessing deterministic. The later API will flag results below 0.60 confidence and ask for a clearer photo or local expert confirmation.
