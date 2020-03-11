$.ajax({
    success: function () {
        loadAllInputOutput();

        //hide mode I/O when it's not selected
        $("#mode").change(function (e) {
            $(`.${e.target.options[
                e.target.selectedIndex
            ].value.toLowerCase()}`).show();
            $(`.${e.target.options[
                1 - parseInt(e.target.selectedIndex)
            ].value.toLowerCase()}`).hide();
        });

        // select embed at first
        $('#mode').val('Embed').change();

        // send image to server side to get preview in return
        $("input.host, input.wm, input.wmed").change(function (e) {
            $.ajax({
                type: 'POST',
                url: `input/${e.target.className}`,
                data: e.target.files[0],
                processData: false,
                contentType: false,
                success: function (address) {
                    $(`img.${e.target.className}`)
                        .attr('src', Flask.url_for("static", { "filename": address }))
                        .width(300)
                        .height(300)
                }
            });
        });

        // click process will start embedding
        $("a.embed.process").click(function (e) {
            $.ajax({
                type: 'POST',
                url: 'embed',
                processData: false,
                contentType: false,
                success: function (response) {
                    $(`img.embed.wmed`)
                        .attr('src', Flask.url_for("static", { "filename": response }))
                        .width(300)
                        .height(300)
                }
            });
        });

        // click process will start extraction
        $("a.extract.process").click(function (e) {
            $.ajax({
                type: 'POST',
                url: 'extract',
                processData: false,
                contentType: false,
                success: function (response) {
                    
                }
            });
        });
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

function getUploadFileButton(type, classname) {
    var div = document.createElement('div');
    var input = setAttributeToElement(
        document.createElement('input'),
        [
            { name: 'type', value: 'file' },
            { name: 'accept', value: 'image/tiff' },
            { name: 'class', value: classname }
        ]
    );
    var btn = setTextToElement(
        setAttributeToElement(
            document
                .createElement('a'),
            [
                { name: 'class', value: `${type} btn btn-primary` }
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

function getCardFor(type, ioType, cardText) {
    let classname = (cardText == 'Watermark') ?
        'wm' :
        (cardText == 'Watermarked') ?
            'wmed' : 'host';

    let form = document.createElement('form');

    let card = setAttributeToElement(
        document
            .createElement("div"),
        [
            { name: 'class', value: `card ${type} ${ioType}` }
        ]
    );
    card.appendChild(
        setAttributeToElement(
            document
                .createElement('img'),
            [
                { name: 'class', value: `card-img-top ${type} ${classname}` }
            ]
        )
    );
    let body = setAttributeToElement(
        document.createElement('div'),
        [{ name: 'class', value: `${type} card-body` }]
    )
    body.appendChild(
        setTextToElement(
            setAttributeToElement(
                document
                    .createElement('h4'),
                [
                    { name: 'class', value: `${type} card-title` }
                ]
            ),
            cardText
        )
    );
    body.appendChild(
        (ioType == 'output') ?
            setTextToElement(
                setAttributeToElement(
                    document
                        .createElement('a'),
                    [
                        { name: 'class', value: `${type} process btn btn-primary` }
                    ]
                ),
                'Process'
            ) :
            getUploadFileButton(type, classname)
    );
    form.appendChild(body);
    card.appendChild(form);
    return document
        .createElement('td')
        .appendChild(card);
}

// only run in document ready

function loadAllInputOutput() {
    loadInputOutputFor(
        'embed',
        [
            ['input', 'Host Image'],
            ['input', 'Watermark'],
            ['output', 'Watermarked']
        ]
    )
    loadInputOutputFor(
        'extract',
        [
            ['input', 'Watermarked'],
            ['output', 'Watermark']
        ]
    );
}

function loadInputOutputFor(type, IO) {
    for (item of IO) {
        document
            .getElementById('files')
            .appendChild(getCardFor(type, item[0], item[1]));
    }
}