$('.multi-field-wrapper').each(function() {
    var $wrapper = $('.multi-fields', this);
  //-> add txtbox and btn
    $(".add-field", $(this)).click(function(e) {
        $('.multi-field:first-child', $wrapper).clone(true).appendTo($wrapper).find('input').val('').focus();
    });

    //-> remove text box and btn
    $('.multi-field .remove-field', $wrapper).click(function() {
        if ($('.multi-field', $wrapper).length > 1)
            $(this).parent('.multi-field').remove();
    });
});

$('#form').submit(function(e) {
e.preventDefault();
 // get all the inputs into an array.
    var $inputs = $('#form :input');
    // not sure if you wanted this, but I thought I'd add it.
    // get an associative array of just the values.
    var values = {};
    let finalArray = []
    let intermediateParams = []
    let intermediateValues=[]
        $inputs.each(function() {
            if (this.name === "params") {
                intermediateParams.push($(this).val())
            } else {
                intermediateValues.push($(this).val())
            }
        });

        data={
            'method':$("#exampleFormControlSelect1").val()
        }
        let filtered = intermediateValues.filter(e => e !== "" )
      for(let i =0; i< intermediateParams.length; i++) {
          let temp = {};
          data[intermediateParams[i]] = filtered[i]
      }
      finalArray['method']=$("#exampleFormControlSelect1").val()

////      console.log("ffff",finalArray)
       $(this).append(finalArray);
        $("#para").val(JSON.stringify(data));

        delete $("#values")

          form = document.getElementById("form"); //$("#frm")
          form.submit();

    return true;
//
//    console.log("url",`${window.location.origin}/model_training/custom_training`)
//        $.ajax({
//                type: "POST",
//                url: `${window.location.origin}/model_training/custom_training`,
//                  data: JSON.stringify({ "hell": finalArray}),
//                    dataType: 'json',
//                contentType:'application/json'
//        }).done(function(data) {
//          console.log("ajax success")
//        })
//        .fail(function(err) {
//          console.error("err", err)
//        })

});

