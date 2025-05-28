# yolov5-deepsort

![Python Version](https://img.shields.io/badge/python-3.9-blue)
![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0.0%2Bcu117-EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white&logoWidth=40)

Questo progetto è un’implementazione open-source di un sistema per il rilevamento e l’allerta dell’attraversamento da parte di utenti deboli della strada (VRU), basato su YOLOv5 per la rilevazione degli oggetti e DeepSORT per il tracciamento multi-oggetto.
L’obiettivo principale è migliorare la sicurezza stradale identificando quando un utente debole della strada (ad esempio un pedone in sedia a rotelle) sta avvicinando o sta tentando di attraversare la strada, generando avvisi tempestivi.
YOLOv5 viene utilizzato per rilevare i VRU in ogni frame video, mentre DeepSORT ne traccia i movimenti nel tempo, assegnando ID univoci.
Il progetto è implementato in Python utilizzando il framework di deep learning PyTorch.


<p align="center">

<img align="center" src="https://github.com/Ayushman-Choudhuri/yolov5-deepsort/blob/main/images/DeepSORT.png">

</p>

## Dipendenze 
Per eseguire il codice, è necessario installare le seguenti dipendenze

* ultralytics 
``` bash
pip install ultralytics
```
* deep-sort-realtime 1.3.2
``` bash
pip install deep-sort-realtime
```
* [pytorch](https://pytorch.org/) - if you want CUDA support with pytorch upgrade the installation based on the CUDA version your system uses. Per avere il supporto di CUDA con pytorch installare la versione più recente sulla base della versione di CUDA utilizzata dal proprio sistema. 

La lista delle dipendenze per il progetto può essere trovata nel file [environment.yml](environment.yml).

Per ricreare l'environment utilizzato per lo sviluppo del codice, è possibile creare un ambiente conda utilizzando il file environment.yml appositamente fornito:
``` bash
conda env create -f environment.yml
```
Questo script creerà un ambiente denominato yolov5-deepsort con tutti i pacchetti necessari installati. Una volta creato, è possibile attivare l'ambiente con il seguente comando:

``` bash
conda activate yolov5-deepsort
```
Una volta attivato l'ambiente, sarà possibile eseguire gli script del progetto e utilizzare i pacchetti elencati in precedenza.


## Struttura delle directory 

```bash
yolov5-deepsort
├── main.py
├── src
    └── dataloader.py
    └── detector.py
    └── tracker.py
    └── utils.py
    └── geom_utils.py
    ├── utils
         └── select_area.py
         ├── coordinates
             └── file.npy
├── models      
├── data
├── outputs
├── environment.yml
├── config.yml
├── README.md


``` 


## Esseguire il progetto
#### Step 1: Assumendo che tutte le dipendenze siano state installate e che l'ambiente sia attivo, clonare la repository.

``` bash
git clone https://github.com/nfloris/vru_crossing_detection

```

### Step 2: Impostare una sorgente video
Accedere al codice sorgente dalla cartella *src* del repository. Dal file **config.yml** è possibile impostare una serie di parametri di configurazione del progetto. Per impostare una sorgente video, è necessario modificare il parametro **input_path** dalla sezione **dataloader**.
Per comodità, tutti i video di input sono posizionati nella cartella *input_videos* del repository.

### Step 2: Selezione geometrica delle aree in prossimità degli attraversamenti pedonali
Per fare in modo che il sistema sia in grado di generare allarmi per pedoni in procinto di attraversare la strada è mnecessario fornire delle informazioni preliminari sulla configurazione dello scenario, specificando le coordinate geografiche delle aree limitrofe agli attraversamenti.
A tal proposito, il file **src/utils/select_area.py** permette, a partire da una schermata, di tracciare manualmente su schermo i contorni geometrici delle aree di interesse. 
Con un click sullo schermo si potranno disegnare delle forme sferiche e ellittiche, editabili tramite i comandi elencati di seguito:

     - a/d -> rotazione verso sinistra/destra
     - w/e -> modifica dell'angolo iniziale/finale (permette di tracciare semicerchi)
     - tasti direzionali -> permettono di spostare l'area lungo l'asse x e y
     - ENTER -> conferma dell'area inserita
     - r -> reset
     - ESC -> chiusura del programma e salvataggio

Una volta terminata l'operazione le informazioni geometriche sulle aree inserite sono salvate in un file di estensione .npy. 
Nella **dataloader** del file **config.yml** è possibile trovare il parametro **area_coordinates_path**, il quale contiene il percorso del file npy contenente le coordinate delle aree specificate per lo scenario di interesse. Per comodità, tutte i file di coordinate sono posizionati all'interno della cartella *src/utils/coordinates* del repository. 


### Step 3: Eseguire il file main.py 
```bash
python3 main.py

```


## Risultati 

Dei risultati di esempio sono consultabili all'interno della cartella *outputs*


## Miglioramenti futuri

1. Sono previsti ulteriori addestramenti dei modelli per migliorare le performance dell'Object Detection.
2. Arricchimento del dataset utilizzato.
3. Nuovi esperimenti su ulteriori utenti deboli della strada.
4. Esperimenti su ulteriori scenari sintetici e esperimenti sul campo.
5. Post-processing dei rilevamenti e dei tracciamenti in modo da migliorare le prestazioni complessive del sistema


   
## References

* [YOLO Algorithm](https://arxiv.org/abs/1506.02640)
* [DeepSORT code repository](https://github.com/nwojke/deep_sort)
* [DATASET]("")

## References

* [YOLO Algorithm](https://arxiv.org/abs/1506.02640)
* [SORT Algorithm](https://arxiv.org/abs/1703.07402)
* [DeepSORT code repository](https://github.com/nwojke/deep_sort)
* [DeepSORT explained](https://medium.com/augmented-startups/deepsort-deep-learning-applied-to-object-tracking-924f59f99104)
