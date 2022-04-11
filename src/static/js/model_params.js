function return_text_input(name,default_value){
    return `<div class="form-group row">
    <h4 for="staticEmail" class="col-sm-3 col-form-label">${name}</h4>
    <div class="col-sm-6" col-md-4>
      <input type="text" name="${name}" id="${name}" class="form-control-plaintext"  value="${default_value}">
    </div>
  </div>`
}

function return_select_input(name,values){
    
    let select=`<select class="form-control" id="${name}" name="${name}">`
    values.forEach((value)=>{
        select+=`<option value="${value}">${value}</option>`
    })
    select+='</select>'
    return `<div class="form-group row">
    <h4 for="staticEmail" class="col-sm-3 col-form-label">${name}</h4>
    <div class="col-sm-6" col-md-4>
      ${select}
    </div>
  </div>`
}

function add_params(patams_obj=[]){
    $("#params").empty();
    patams_obj.forEach((element)=>{
        let comp_=''
        if(element.type==='select'){
             comp_=return_select_input(element.name,element.values)
        }else{
            comp_=return_text_input(element.name,element.values)
        }
        $("#params").append(comp_);
    })
}