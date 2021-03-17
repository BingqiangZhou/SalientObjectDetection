# mkdir ./sod_dataset

cd ./sod_dataset
# mkdir images
# mkdir gts

# wget http://saliencydetection.net/dut-omron/download/DUT-OMRON-image.zip
# wget http://saliencydetection.net/dut-omron/download/DUT-OMRON-gt-pixelwise.zip.zip
# wget http://mftp.mmcheng.net/Data/MSRA-B.zip

unzip -q ./DUT-OMRON-image.zip
unzip -q ./DUT-OMRON-gt-pixelwise.zip.zip

mv ./DUT-OMRON-image/*.jpg ./images
mv ./pixelwiseGT-new-PNG/*.png ./gts

# unzip -q ./MSRA-B.zip

# mv ./MSRA-B/*.jpg ./images
# mv ./MSRA-B/*.png ./gts