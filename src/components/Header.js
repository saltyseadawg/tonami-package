import React, { Component } from 'react';
import styles from '../SCSS/Main.module.scss';

export default class Header extends Component{
  constructor( props ) {
    super();

    this.handleOnClick = this.handleOnClick.bind( this );
  }

  handleOnClick( id ) {
    if ( id === 'home' ) this.props.history.push('/');
    if ( !document.getElementById( id ) ) return null;
    document.getElementById( id ).scrollIntoView({ behavior: "smooth" });
  }

  componentDidMount() {
    window.addEventListener('scroll', this.handleScroll);
  }

  componentWillUnmount() {
    window.removeEventListener('scroll', this.handleScroll);
  }

  handleScroll(event) {
    var navBar = document.getElementById("myNavbar");
    var navDemo = document.getElementById("navDemo");
      if ( !document.body.scrollTop || !document.documentElement.scrollTop ) {
          navBar.className = `${ styles.bar } ${ styles.desktop } scroll`;
          navDemo.className = `${ styles.bar } ${ styles.mobile } scroll`;
      } else {
          navBar.className = `${ styles.bar } ${ styles.desktop } noScroll`;
          navDemo.className = `${ styles.bar } ${ styles.mobile } noScroll`;
      }
  }

  render(){
    if ( this.props.isHome )
      return(
        <WidthHandleLayout>
          <div className={ styles.bar } mobileonly="true">
            <div className={ styles.barItem } onClick={ () => this.handleOnClick( 'home' ) }><i className="fa fa-home" /></div>
            <ThemeToggle theme={ this.props.state.theme } setState={ this.props.setState } />
          </div>
          <div className={ styles.bar } desktoponly="true">
            <div className={ styles.barItem } onClick={ () => this.handleOnClick( 'home' ) }><i className="fa fa-home" /> HOME</div>
            <div style={{flex: 1}} />
            <ThemeToggle theme={ this.props.state.theme } setState={ this.props.setState } />
          </div>
        </WidthHandleLayout>
      )
    return(
      <WidthHandleLayout>
        <div className={ styles.bar }>
          <div style={{ width: "8em", textAlign: "right"}}><div className={ styles.barItem } onClick={ () => this.handleOnClick( 'home' ) }><i className="fa fa-home" /><div className={ styles.iconText }> HOME</div></div></div>
          <div style={{flex: 1}} />
          <ThemeToggle theme={ this.props.state.theme } setState={ this.props.setState } />
        </div>
      </WidthHandleLayout>
    )
  }
}

class WidthHandleLayout extends Component {
  render(){
    return(
      <div className={ `${ styles.header } header` }>
        <div className={ `${ styles.bar } ${ styles.desktop } scroll` } id="myNavbar">
          {React.Children.map( this.props.children, child => (
              React.cloneElement(child, { style: child.props.mobileonly ? { display: "none" } : { ...child.props.style }})
          ))}
        </div>
        <div className={ `${ styles.bar } ${ styles.mobile } noScroll`} id="navDemo">
          {React.Children.map( this.props.children, child => (
              React.cloneElement(child, { style: child.props.desktoponly ? { display: "none" } : { ...child.props.style }})
          ))}
        </div>
      </div>
    )
  }
}

class ThemeToggle extends Component {
  render(){
    return( 
      <div style={{ width: "8em", textAlign: "right"}}>
        { this.props.theme === "default"
          ? <div className={ styles.barItem } style={{ float: "right" }} onClick={ () => this.props.setState({ theme: "dark" })}><i className="fa fa-moon" /><div className={ styles.iconText }> DARK</div></div>
          : <div className={ styles.barItem } style={{ float: "right" }} onClick={ () => this.props.setState({ theme: "default" })}><i className="fa fa-sun" /><div className={ styles.iconText }> DEFAULT</div></div>
        }
      </div>
    )
  }
}
