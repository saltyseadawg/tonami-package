import React, { Component } from 'react';
import styles from '../SCSS/Main.module.scss';
import stylesTheme from '../SCSS/App.scss';
import { HomePageSections } from './Constants';

export default class Header extends Component{
  constructor( props ) {
    super();
    this.state={
      mobileOpen: false,
      projectOpen: false,
      projectList: props.state.projectList,
      i: props.i,
    }
    
    this.handleOnClick = this.handleOnClick.bind( this );
  }

  handleOnClick( id ) {
    this.setState({mobileOpen: false });
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

  componentDidUpdate( prevProps ){
    if ( !prevProps.state.projectList.length ) 
      this.setState({projectList: this.props.state.projectList})
    if ( prevProps.i !== this.props.i )
      this.setState({i: this.props.i})
  }

  render(){
    if ( this.props.isHome )
      return(
        <WidthHandleLayout>
          <div className={ styles.bar } mobileonly="true">
            <div className={ styles.barItem } onClick={ () => this.handleOnClick( 'home' ) }><i className="fa fa-home" /></div>
            <div className={ styles.barItem } style={{flex: 1}} onClick={ () => this.setState({mobileOpen: !this.state.mobileOpen }) }><i className={ this.state.open ? "fa fa-chevron-up" : "fa fa-chevron-down" }/></div>
            <ThemeToggle theme={ this.props.state.theme } setState={ this.props.setState } />
          </div>
          <div className={ styles.bar } desktoponly="true">
            <div className={ styles.barItem } onClick={ () => this.handleOnClick( 'home' ) }><i className="fa fa-home" /> HOME</div>
            <HomePageSectionsComponent handleOnClick={ this.handleOnClick } />
            <div style={{flex: 1}} />
            <ThemeToggle theme={ this.props.state.theme } setState={ this.props.setState } />
          </div>
          <div className={ styles.bar }>
            { this.state.mobileOpen && <HomePageSectionsComponent handleOnClick={ this.handleOnClick } /> }
          </div>
        </WidthHandleLayout>
      )
    const first = !this.state.projectList[0] ? false : !this.state.i;
    const last = !this.state.projectList[0] ? false : this.state.i === this.state.projectList.length -1;
    return(
      <WidthHandleLayout>
        <div className={ styles.bar }>
          <div style={{ width: "8em", textAlign: "right"}}><div className={ styles.barItem } onClick={ () => this.handleOnClick( 'home' ) }><i className="fa fa-home" /><div className={ styles.iconText }> HOME</div></div></div>
          <div style={{flex: 1}} />
          <ProjectNav toggleProjectOpen={ () => this.setState({ projectOpen: !this.state.projectOpen })} history={ this.props.history } projectList={ this.state.projectList } i={ this.state.i } first={ first } last={ last } />
          <div style={{flex: 1}} />
          <ThemeToggle theme={ this.props.state.theme } setState={ this.props.setState } />
        </div>
        <ProjectOpen projectOpen={ this.state.projectOpen } history={ this.props.history } projectList={ this.state.projectList } />
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

class HomePageSectionsComponent extends Component {
  render(){
    return(
      <React.Fragment>
        { HomePageSections.map( ( section ) =>
          <div className={ styles.barItem } onClick={ () => this.props.handleOnClick( section.title ) } key={ section.title }><i className={ section.icon } /> { section.title }</div>
        ) }
      </React.Fragment>
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

class ProjectNav extends Component {
  render(){
    return(
      <div style={{ display: "flex" }}>
        <div onClick={ () => this.props.history.push( this.props.first ? "/" : ("#page" + this.props.projectList[ this.props.i - 1 ].projectName ) ) } style={ this.props.first ? { visibility: "hidden"} : {color: "inherit"}}>
          <div className={ styles.barItem }><i className="fa fa-chevron-left" /></div>
        </div>
        <div className={ styles.barItem } onClick={ () => this.props.toggleProjectOpen() } ><i className="fa fa-bars"/> PROJECT LIST</div>
        <div onClick={ () => this.props.history.push( this.props.last ? "/" : ("#page" + this.props.projectList[ this.props.i + 1 ].projectName ) ) } style={ this.props.last ? { visibility: "hidden"} : {color: "inherit"}}>
          <div className={ styles.barItem }><i className="fa fa-chevron-right" /></div>
        </div>
      </div>
    )
  }
}

class ProjectOpen extends Component {
  render() {
    return(
      <div className={ styles.projectOpen } style={ this.props.projectOpen ? {} : { display: "none"}}>
        { this.props.projectList.map((project, i) => 
          <div onClick={ () => this.props.history.push( "/#page" + project.projectName ) } style={{color: "inherit"}} key={ i }>
            <div className={ styles.barItem }>
              <img src={ require('../images/' + project.thumbnail.image + '.jpg' ) } alt={ project.thumbnail.alt } />
              {project.thumbnail.name}
            </div>
          </div>
        )}
      </div>
    )
  }
}