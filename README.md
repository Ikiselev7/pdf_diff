## Quickstart

* Ensure you have [Python 3.8+](https://www.python.org/downloads/), [Node.js](https://nodejs.org), and [npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) and [Yarn](https://www.npmjs.com/package/yarn) installed.
* Clone this repo.
* Create a new Python virtual environment for the template:
```
$ cd pdf_diff
$ python3 -m venv .venv  # create venv
$ source .venv/bin/activate   # activate venv
$ python -m spacy download en_core_web_lg
$ pip install -r requirements.txt # install streamlit
```
* Add yolo dependencies:
```
$ pip install yolov5_dh/requirements.txt
```
* Initialize and run the component template frontend:
```
$ cd pdf_diff/diff/frontend
$ yarn install    # Install npm dependencies
$ yarn start  # Start the Webpack dev server
```
* From a separate terminal, run the template's Streamlit app:
```
$ cd pdf_diff
$ . venv/bin/activate  # activate the venv you created earlier
$ export PYTHONPATH='./:./yolov5_dh'
$ streamlit run diff/app.py  # run the example
```