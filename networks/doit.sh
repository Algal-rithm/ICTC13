#!/bin/bash
echo "Starting the test"

resultsFile="MORPH_448_test3.txt" # change as needed

# *example*
## relates to this "python3 Algal-rithim_1.py -f 0 -g 0 -m 1 -o sgd -e 20 -a /home/researcher/exp1/PCC_73104 -b /home/researcher/exp1/CCAP_140313E -d /home/researcher/exp1"

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 0 -o sgd1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 1 -o sgd1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 2 -o sgd1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 7 -o sgd1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 3 -o sgd1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 4 -o sgd1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 5 -o sgd1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 6 -o sgd1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 8 -o sgd1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 0 -o adam1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 1 -o adam1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 2 -o adam1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 7 -o adam1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 3 -o adam1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 4 -o adam1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 5 -o adam1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 6 -o adam1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 Algal-rithim_1.py -f 4 -g 0 -m 8 -o adam1 -e 100 -a /home/researcher/exp1/NodHar_1 -b /home/researcher/exp1/NodHar_73104 -d /home/researcher/exp1 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}





