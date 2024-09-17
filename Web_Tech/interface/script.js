let navbar = document.querySelector('.navbar');

document.querySelector('#menu-btn').onclick = () =>{
    navbar.classList.toggle('active');
    searchForm.classList.remove('active');
    cartItem.classList.remove('active');
}

let searchForm =document.querySelector('.search-form');

focument.querySelector('#search-btn').onclick = () =>{
    cartItem.classList.toggle('active');
    navbar.claasList.remove('active');
    searchForm.classList.remove('active');
}


let cartItem=document.querySelector('.cart-items-container');

focument.querySelector('#cart-btn').onclick = () =>{
    cartItem.classList.toggle('active');
    navbar.claasList.remove('active');
    searchForm.classList.remove('active');
    
}


window.onscroll= () =>{
    navbar.claasList.remove('active');
    searchForm.classList.remove('active');
    cartItem.classList.remove('active');
}