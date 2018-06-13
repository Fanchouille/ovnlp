import requests
import json
import os
import zipfile, gzip, shutil
from tqdm import tqdm
import time
from pathlib import Path
import pkg_resources
from gensim.models.fasttext import FastText


def get_data(iUrl):
    """
    Get data from URL
    :param iUrl:
    :return:
    """
    try:
        r = requests.get(iUrl, stream=True)
        return r
    except:
        print("Unable to fetch data source.")
        raise


def get_dir(dirPath):
    deleteFiles = []
    deleteDirs = []
    for root, dirs, files in os.walk(dirPath):
        for f in files:
            deleteFiles.append(os.path.join(root, f))
        for d in dirs:
            deleteDirs.append(os.path.join(root, d))
    return deleteFiles, deleteDirs


def delete_dir(dirPath):
    if os.path.exists(dirPath):
        deleteFiles = []
        deleteDirs = []
        for root, dirs, files in os.walk(dirPath):
            for f in files:
                deleteFiles.append(os.path.join(root, f))
            for d in dirs:
                deleteDirs.append(os.path.join(root, d))
        for f in deleteFiles:
            os.remove(f)
        for d in deleteDirs:
            os.rmdir(d)
        os.rmdir(dirPath)
    else:
        pass


class WeightSource(object):
    def __init__(self, iLang=None, iTrainedOn=None, iSavePath=None):
        """

        :param iDict: dict of pretrained weight source (weightsource.json)
        :param iTrainedOn: cc (common crawl) or wiki
        :param iSavePath:
        """
        self.dsDict = json.loads(pkg_resources.resource_string(__name__, 'weightsource.json'))

        if iLang is None:
            print("Non language specified, 'en' by default.")
            self.lang = "en"
        else:
            self.lang = iLang

        if iTrainedOn is not None:
            if iTrainedOn not in self.dsDict:
                print("-" * 100)
                print("Custom model  : " + iTrainedOn + "")
                print("-" * 100)
                self.trainedOn = iTrainedOn
            else:
                print("-" * 100)
                print(
                    "Pretrained weights are provided by FB for * " + iTrainedOn + " *.\nPlease use another name if you want to create your own model.")
                print("-" * 100)
                self.trainedOn = iTrainedOn
        else:
            print("-" * 100)
            print("Custom model : myModel.")
            print("-" * 100)
            self.trainedOn = "myModel"

        if iSavePath is None:
            self.projectPath = str(Path.home()) + "/ovnlp"
        else:
            self.projectPath = iSavePath

        self.sourceDict = self.get_weight_source_dict()

    def get_weight_source_dict(self):
        """

        :param iLang:
        :return:
        """
        if self.trainedOn in self.dsDict:
            if self.lang in self.dsDict[self.trainedOn]:
                lUrl = self.dsDict[self.trainedOn][self.lang]
                lSmallPath = self.projectPath + "/fasttext/weights/" + self.trainedOn + "/" + self.lang
                lFilename = lUrl.split("/")[::-1][0]
                lFullPath = lSmallPath + '/' + lFilename
                lZipFormat = lFilename.split(".")[::-1][0]
                lProjectPath = self.projectPath
                return {"url": lUrl, "smallPath": lSmallPath, "fullPath": lFullPath, "filename": lFilename,
                        "zipformat": lZipFormat, "projectPath": lProjectPath}
            else:
                print("Language " + self.lang + " is not available.")
        else:
            lSmallPath = self.projectPath + "/fasttext/weights/" + self.trainedOn + "/" + self.lang
            lFilename = self.trainedOn + "." + self.lang + ".bin"
            lFullPath = lSmallPath + '/' + lFilename
            lProjectPath = self.projectPath
            return {"smallPath": lSmallPath, "fullPath": lFullPath, "filename": lFilename, "projectPath": lProjectPath}

    def dl_weights(self, iResave=False):
        """"""
        if (not os.path.exists(self.sourceDict["smallPath"])) | iResave:
            delete_dir(self.sourceDict["smallPath"])

            os.makedirs(self.sourceDict["smallPath"])

            r = get_data(self.sourceDict["url"])
            block_size = 1024
            total_size = int(r.headers.get('content-length', 0))

            print("*" * 100)
            print("Downloading " + self.sourceDict["filename"] + " file from fastText")
            print("*" * 100)
            time.sleep(1)
            pbar = tqdm(total=total_size, initial=0, unit='B', unit_scale=True)
            with open(self.sourceDict["fullPath"], "wb") as handle:
                for chunk in r.iter_content(block_size):
                    if chunk:
                        handle.write(chunk)
                        pbar.update(block_size)
            print(self.lang + " weights saved at " + self.sourceDict["fullPath"])

        else:
            pass

    def extract_weights(self):
        if len(get_dir(self.sourceDict["smallPath"])[0]) > 0:
            if \
                    [file.split(".") for file in get_dir(self.sourceDict["smallPath"])[0] if ".DS_Store" not in file][
                        0][::-1][
                        0] == "bin":
                print("Weights already downloaded and extracted in " + self.sourceDict["smallPath"] + ".")
            else:
                if self.sourceDict["zipformat"] == "zip":
                    try:
                        z = zipfile.ZipFile(self.sourceDict["fullPath"])
                        z.extractall(self.sourceDict["smallPath"])
                        print("Zip files successfully extracted.")
                    except:
                        print(
                            "Weights zip file seems to be corrupted.\nPlease launch dl_weights(iResave=True) to redownload weights file.")
                elif self.sourceDict["zipformat"] == "gz":
                    try:
                        with gzip.open(self.sourceDict["fullPath"], 'rb') as f_in:
                            with open(self.sourceDict["fullPath"].replace("." + self.sourceDict["zipformat"], ""),
                                      'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)

                        print("GZip files successfully extracted.")

                    except:
                        print(
                            "Weights zip file seems to be corrupted.\nPlease launch dl_weights(iResave=True) to redownload weights file.")

                else:
                    print("Zip file format unknown : please check zip format is .zip or .gz")
        else:
            if (not os.path.exists(self.sourceDict["fullPath"])):
                print("Please download weights first by using dl_weights(\"" + self.lang + "\")")

    def delete_archive(self):
        if (os.path.exists(self.sourceDict["fullPath"])):
            os.remove(self.sourceDict["fullPath"])
        else:
            pass

    def save_weights(self, iTrainedModel=None, iResave=False):
        if self.trainedOn in self.dsDict:
            self.dl_weights(iResave)
            self.extract_weights()
            self.delete_archive()

        else:
            if iTrainedModel is None:
                print("Please provide a trained model to save it.")
                return
            else:
                if (not os.path.exists(self.sourceDict["smallPath"])) | iResave:
                    delete_dir(self.sourceDict["smallPath"])
                    os.makedirs(self.sourceDict["smallPath"])
                    iTrainedModel.save(self.sourceDict["fullPath"])
                    print("Custom model saved in " + self.sourceDict["fullPath"])
                else:
                    print("Weights already trained. Please use iResave=True to resave.")
                    return

    def load_model(self):
        files, dirs = get_dir(self.sourceDict["smallPath"])
        fileToLoad = [file for file in files if ".bin" in file]
        if len(fileToLoad) > 0:
            try:
                print("Loading with FastText.load")
                model = FastText.load(fileToLoad[0])
                print(fileToLoad[0] + " was loaded.")
            except:
                try:
                    print("Loading with FastText.load_fasttext_format")
                    model = FastText.load_fasttext_format(fileToLoad[0])
                    print(fileToLoad[0] + " was loaded.")
                except:
                    print("Unable to load " + fileToLoad[0] + " file. Please retrain or redl weights.")

            return model
        else:
            print("No model found. Please dl pretrained weights or train custom ones.")
            return


def train_weights(iSentencesList, **kwargs):
    if kwargs is not None:
        model = FastText(**kwargs)
    else:
        model = FastText()
    model.build_vocab(iSentencesList)
    model.train(iSentencesList, total_examples=model.corpus_count, epochs=model.epochs)
    print("Custom model trained.")
    return model


def main():
    # Instantiate WeightSource class
    ws = WeightSource(iTrainedOn="cc", iLang="fr")
    # Dl and extract weights for language = "fr" for instance
    ws.save_weights(iResave=False)
    return


if __name__ == "__main__":
    main()
