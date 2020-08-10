Instalar Python 3
Rodar o comando abaixo na mesma pasta do arquivo Outros/requirements.txt
pip install -r requirements.txt

Os fontes estão separados da seguinte forma:

Fontes/scrapping
Contém os scripts para extrair as imagens e nível do rio Itajaí-Açú da Defesa Civil de Rio do Sul-SC

Fontes/mask_to_json
Após utilizar a ferramenta Labelbox para segmentar a silhueta do rio, nessa pasta contém os scripts para separar as imagens em pastas de treino (dataset.py e annotate.py), teste e validação, e contém também o script (mask2json.py) para extrair as coordenadas da segmentação em um arquivo json com os pontos xy.

Fontes/training
Contém os scripts para o treinamento da Mask R-CNN (river.py) e para a medição do nível do rio (evaluate.py e level_calculator.py).


Para utilizar:
Utilizar os scripts de scrapping para extrair as imagens.
Utilizar os scripts dataset.py e annotate.py para separar as imagens e em seguida o mask2json.py para extrair as coordenadas.
Utilizar o script river.py para treinar a Mask R-CNN com base nas imagens separadas.
Utilizar o script evaluate.py e level_calculator.py para calbirar a medição do nível e inferir e disponibilizar o nível.

