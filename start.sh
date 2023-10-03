#! /bin/bash

python3 thesh-images.py png/ processed-pngs/ -pdf output_pdfs/diamante_revised_cleaned.pdf




python3 thesh-images.py png/ processed-pngs-x/
python3 processed-pngs-x/ -pdf output_pdfs/diamante_revised_cleaned.pdf


python3 thesh-images.py png processed-pngs-x -o 53 -n 1 --rescale_factor 1.0
python3 thesh-images.py processed-pngs-x output_pdfs/diamante_revised_cleaned.pdf --disable_processing

python3 thesh-images.py png/ processed-pngs-full/ -pdf output_pdfs/diamante_revised_cleaned_full.pdf -o 60 -n 3 --compress --output_format 'jpg' --jpg_quality 80 --add_parameters --create_pdf_dir

#For serious run
python3 thesh-images.py png/ processed-pngs-full/ -pdf output_pdfs/diamante_revised_cleaned_full.pdf --shift -o 60 -n 3 --compress --output_format 'jpg' --jpg_quality 80 --add_parameters --create_pdf_dir

## Full test run (Note: rescaling happens before processing which improves runtime, but also causes a form aliasing after the processing)
python3 thesh-images.py png processed-pngs-full -pdf output_pdfs/diamante_revised_cleaned_full.pdf --output_format 'png' --jpg_quality 100 --add_parameters --create_pdf_dir -rs 0.5 --compress -o 80 -n 5

## 2 Step run
python3 thesh-images.py png processed-pngs-2step-test --compress -o 90 -n 10 && \
python3 thesh-images.py processed-pngs-2step-test output_pdfs/2-step-test.pdf --compress --add_parameters --output_format 'png' & \
python3 thesh-images.py processed-pngs-2step-test output_pdfs/2-step-test.pdf --compress --add_parameters --output_format 'jpg' --jpg_quality 75 & \
python3 thesh-images.py processed-pngs-2step-test output_pdfs/2-step-test.pdf --compress --add_parameters --output_format 'jpg' --jpg_quality 50 & \
python3 thesh-images.py processed-pngs-2step-test output_pdfs/2-step-test.pdf --compress --add_parameters --output_format 'png' -rs 0.5 & \
python3 thesh-images.py processed-pngs-2step-test output_pdfs/2-step-test.pdf --compress --add_parameters --output_format 'jpg' --jpg_quality 75 -rs 0.5 & \
python3 thesh-images.py processed-pngs-2step-test output_pdfs/2-step-test.pdf --compress --add_parameters --output_format 'jpg' --jpg_quality 50 -rs 0.5 & \
python3 thesh-images.py processed-pngs-2step-test output_pdfs/2-step-test.pdf --compress --add_parameters --output_format 'png' -rs 0.4 & \
python3 thesh-images.py processed-pngs-2step-test output_pdfs/2-step-test.pdf --compress --add_parameters --output_format 'png' -rs 0.3 & \
python3 thesh-images.py processed-pngs-2step-test output_pdfs/2-step-test.pdf --compress --add_parameters --output_format 'png' -rs 0.2


### Diamante
python3 thesh-images.py png processed-diamante-pngs --compress --shift -t 10
python3 thesh-images.py processed-diamante-pngs output_pdfs/diamante_revised_cleaned_full.pdf --compress --add_parameters --output_format 'png'

python3 thesh-images.py processed-diamante-pngs output_pdfs/diamante_revised_cleaned_full.pdf --compress --add_parameters --output_format 'png' --save_pdf_matrix --threads 8