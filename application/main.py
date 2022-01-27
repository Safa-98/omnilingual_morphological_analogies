from flask import Flask
from flask import request, escape, render_template, jsonify
from waitress import serve
from modules.nn_analogy_solver.solver import Solver
from modules.interface.interface import Interface


app = Flask(__name__,template_folder='templates')

interface = Interface()

@app.route('/', methods=['GET', 'POST'])
def app_home():
    '''Displays the home page.
    '''
    return render_template('index.html', result_text = "", word_d_background = "#fff", example_features="")

@app.route('/solveAnalogy',methods=["GET", 'POST'])
def solveAnalogy():
    '''Solve the analogy on screen.
    '''
    print('Solve Analogy')
    word_a = request.form.get("word_a")
    word_b = request.form.get("word_b")
    word_c = request.form.get("word_c")
    word_d = request.form.get("word_d")

    interface.A = word_a if word_a else None
    interface.B = word_b if word_b else None
    interface.C = word_c if word_c else None
    print(f"\tCurrent example: {interface.A}, {interface.B}, {interface.C}")
    result = interface.solve()
    print(f"\tResult: {result}\n")

    if interface.check_example_exists(word_a, word_b, word_c, word_d if word_d else None):

        if interface.D == result:
            message = ""
            word_d_background = "#52FF79"
        else:
            message = f"This result is not the expected one ({interface.D})."
            word_d_background = "#FF5252"

    else:
        interface.D = result
        message = f"This analogy is not in our database, the result might be unexpected."
        word_d_background = f"#BBBBBB"

    output = {'word_d': result,
                'result': message,
                'word_d_background': word_d_background,
                'word_d_shadow': f"0px 0px 3px 3px {word_d_background}"}
    return jsonify(output)


@app.route("/giveExample",methods=["GET", 'POST'])
def giveExample():
    '''Displays an example of analogy.
    '''
    print('Get an example')

    word_a = request.form.get("word_a")
    word_b = request.form.get("word_b")
    word_c = request.form.get("word_c")
    source_language = request.form.get("source_language").lower()
    target_language = request.form.get("target_language").lower()
    selected_features = request.form.get("selected_features")

    possible_features = interface.get_features_list(selected_features)
    print(f"\t{0 if possible_features is None else len(possible_features)} such set{'' if (possible_features is None or len(possible_features) <2) else 's'} of features")

    if possible_features is not None and not len(possible_features):
        print("\t! No such features !\n")
        interface.A = ""
        interface.B = ""
        interface.C = ""
        features = ""
        message = "Please choose a valid set of features."
    else:
        interface.A = word_a if word_a else None
        interface.B = word_b if word_b else None
        interface.C = word_c if word_c else None
        interface.source_language = source_language if source_language != 'any language' else None
        interface.target_language = target_language if target_language != 'any language' else None
        interface.features = selected_features if selected_features != "" else None

        print(f"\tPrevious example: {interface.A}, {interface.B}, {interface.C}")
        print(f"\tLanguages: {interface.source_language.capitalize() if interface.source_language is not None else 'Any language'} -> {interface.target_language.capitalize() if interface.target_language is not None else 'Any language'}")
        print(f"\tFeatures: {selected_features if selected_features != '' else 'None'}")
        example = interface.get_example(possible_features)

        if example is not None:
            interface.A = example['A'].values[0]
            interface.B = example['B'].values[0]
            interface.C = example['C'].values[0]
            interface.D = example['D'].values[0]
            features = f"<b>{example['source_language'].values[0].capitalize()} to {example['target_language'].values[0].capitalize()}</b>: {example['features'].values[0]}"#"Features: " + example['features'].values[0]
            print(f"\tNew example: {interface.A}, {interface.B}, {interface.C}, {interface.D}")
            print(f"\tNew languages: {example['source_language'].values[0]} -> {example['target_language'].values[0]}\n")
            message = ""
        else:
            print("\t! No example with these options !\n")
            features = ""
            message = "No example with these options"

    output = {'word_a': interface.A,
                'word_b': interface.B,
                'word_c': interface.C,
                'features': features,
                'result': message}
    return jsonify(output)


@app.route('/shuffleWords',methods=["GET", 'POST'])
def shuffleWords():
    '''Shuffle the analogy on screen.
    '''
    print('Shuffle Analogy')
    print(f"\tLanguages: {interface.source_language.capitalize() if interface.source_language is not None else 'Any language'} -> {interface.target_language if interface.target_language is not None else 'Any language'}")
    word_a = request.form.get("word_a")
    word_b = request.form.get("word_b")
    word_c = request.form.get("word_c")
    word_d = request.form.get("word_d")

    interface.A = word_a if word_a else None
    interface.B = word_b if word_b else None
    interface.C = word_c if word_c else None
    print(f"\tPrevious example: {interface.A}, {interface.B}, {interface.C}, {interface.D}")
    interface.shuffle()
    print(f"\tNew example: {interface.A}, {interface.B}, {interface.C} (, {interface.D})\n")

    message = ""

    output = {'word_a': interface.A,
                'word_b': interface.B,
                'word_c': interface.C,
                'word_d': interface.D if (word_d is not None and word_d) else "",
                'source_language': interface.source_language.capitalize() if interface.source_language is not None else 'Any language',
                'target_language': interface.target_language.capitalize() if interface.target_language is not None else 'Any language',
                'result': message}
    return jsonify(output)

@app.route('/updateFeatures',methods=["GET", 'POST'])
def updateFeatures():
    '''Update the list of possible features.
    '''
    source_language = request.form.get("source_language").lower()
    target_language = request.form.get("target_language").lower()
    selected_features = request.form.get("selected_features").lower()
    print(f'Update features list to ({source_language} to {target_language}) ones:')

    interface.source_language = source_language if source_language != 'any language' else None
    interface.target_language = target_language if target_language != 'any language' else None
    features_list, nb_features = interface.get_possible_features()
    print(f"\t{nb_features} set{'s' if nb_features > 1 else ''} of features now\n")
    output = {'features_list': features_list}
    return jsonify(output)


if __name__ == "__main__":
    if app.config['DEBUG'] == True:
            app.run(debug=True)
    else:
        host = '0.0.0.0'
        port = 5000
        print(f"Launch the app on http://{host}:{port}")
        serve(app, host=host, port=port)

