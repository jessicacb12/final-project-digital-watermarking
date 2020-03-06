$.ajax({
    success: function () {
        loadInputOutput([
            ['input', 'Host Image'],
            ['input', 'Watermark'],
            ['output', 'Watermarked']
        ]);
    }
});

// Can-be-used-by-all functions

function setAttributeToElement(element, attributes) {
    attributes.forEach(attr => {
        var attrHTML = document.createAttribute(attr.name);
        attrHTML.value = attr.value;
        element.setAttributeNode(attrHTML);
    });
    return element;
}

function setTextToElement(element, text) {
    element.appendChild(
        document.createTextNode(text)
    );
    return element;
}

// UI files functions

function displayInput(input) {
    if (input.files && input.files[0]) {
        let reader = new FileReader();

        reader.onload = function (e) {
            $(`img.${input.classList[0]}`)
                .attr('src', e.target.result)
                .width(300)
                .height(300)
        }

        reader.readAsDataURL(input.files[0]);
    }
}

function getUploadFileButton(classname) {
    var div = document.createElement('div');
    var input = setAttributeToElement(
        document.createElement('input'),
        [
            { name: 'type', value: 'file' },
            { name: 'accept', value: 'image/tiff' },
            { name: 'class', value: classname }
            // { name: 'onchange', value: '' }
        ]
    );
    var btn = setTextToElement(
        setAttributeToElement(
            document
                .createElement('a'),
            [
                { name: 'href', value: '#' },
                { name: 'class', value: 'btn btn-primary' }
            ]
        ),
        'Upload'
    );

    btn.addEventListener('click', function () {
        input.click();
    });

    div.appendChild(input);
    div.appendChild(btn);
    return div;
}

//hardcoded card form

function getCardFor(type, cardText) {
    let classname = (cardText == 'Watermark') ?
                         'wm' :
                         (cardText == 'Watermarked')?
                         'wmed' : 'host';

    let card = setAttributeToElement(
        document
            .createElement("div"),
        [
            { name: 'class', value: `card ${type}` }
        ]
    );
    card.appendChild(
        setAttributeToElement(
            document
                .createElement('img'),
            [
                { name: 'class', value: `card-img-top ${classname}` }
            ]
        )
    );
    let body = setAttributeToElement(
        document.createElement('div'),
        [{ name: 'class', value: 'card-body' }]
    )
    body.appendChild(
        setTextToElement(
            setAttributeToElement(
                document
                    .createElement('h4'),
                [
                    { name: 'class', value: 'card-title' }
                ]
            ),
            cardText
        )
    );
    body.appendChild(
        (type == 'output') ?
            setTextToElement(
                setAttributeToElement(
                    document
                        .createElement('a'),
                    [
                        { name: 'href', value: '#' },
                        { name: 'class', value: 'btn btn-primary' }
                    ]
                ),
                'Process'
            ) :
            getUploadFileButton(classname)
    );
    card.appendChild(body);
    return document
        .createElement('td')
        .appendChild(card);
}

function changeInputOutput() {
    // let mode = document.getElementById('mode');
    (document.getElementById('mode').value == 'Embed') ?
        loadInputOutput([
            ['input', 'Host Image'],
            ['input', 'Watermark'],
            ['output', 'Watermarked']
        ]) :
        loadInputOutput([
            ['input', 'Watermarked'],
            ['output', 'Watermark']
        ]);
}

function removePreviousInputOutput() {
    $('.input').remove();
    $('.output').remove();
}

//hardcoded 2 input 1 output for host image, watermark, watermarked

function loadInputOutput(IO) {
    removePreviousInputOutput();
    for (item of IO) {
        document
            .getElementById('files')
            .appendChild(getCardFor(item[0], item[1]));
    }
}