import React, { Component } from 'react';
import Header from './Header';

export class MainLayout extends Component{
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
        </div>
      </div>
      );
  }
}
