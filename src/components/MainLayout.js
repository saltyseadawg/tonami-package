import React, { Component } from 'react';
import Header from './Header';
import Footer from './Footer';
import styles from '../SCSS/Main.module.scss';

export class MainLayout extends Component{
  constructor( props ) {
    super();

    this.handleTop = this.handleTop.bind( this );
  }
  handleTop() {
    document.getElementById('main').scrollIntoView({ behavior: "smooth" });
  }

  render(){
    var index = window.location.href.indexOf("#page");
    if ( index !== -1 )
      this.props.history.push( '/' + window.location.href.slice( index + 5 ) )
    return(
      <div className={ "app-defaults theme-" + this.props.state.theme } id="main">
        <div className="app-container">
          <Header {...this.props} />
          { React.Children.map( this.props.children, child =>
              React.cloneElement( child ),)}
          <div className={ styles.toTop } onClick={ () => this.handleTop() }><i className="fas fa-angle-double-up" /></div>
          <Footer {...this.props} />
        </div>
      </div>
      );
  }
}
