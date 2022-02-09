import React, { Component } from 'react';
import stylesHome from '../SCSS/HomePage.module.scss';

export default class Exercise2 extends Component{
  constructor() {
    super();

    var source=require('../JSONs/HomePage.json');

    this.state={
      source: source
    }

  }

  render(){
    var source = this.state.source;
    return(
      <InterestsSection interests={ source.interests } />
    );
  }
}

class InterestsSection extends Component {
  state={
    hover: undefined,
    selectedInterest: 0
  }
  render(){
    var openInterest = this.props.interests[ this.state.selectedInterest ];
    return(
      <div className={ stylesHome.sectionContainer} id="interests">
        <h3>OUTSIDE OF OFFICE HOURS</h3>
        <div className={ stylesHome.subTitle }>Conversation Starters <br />Things Nobody Asked For<br />Weak Flexes</div>
        <div className={ stylesHome.interestsWrapper }>
          <div className={ stylesHome.interestsBar }>
            { this.props.interests.map( ( interest, i ) =>
              <div key={ i } className={ stylesHome.option } onClick={ () => this.setState({ selectedInterest: i }) } onMouseEnter={ () => this.setState({ hover: i })} onMouseLeave={ () => this.setState({ hover: undefined })} style={ ( this.state.hover !== undefined && i!== this.state.hover ) ? {opacity: 0.4} : {opacity: 1}}>
                <img src={ require('../images/' + interest.image + '.jpg' ) } style={{width: "20%", height: "100%", objectFit: "contain", minWidth: "4em", padding: "5px"}} alt={ interest.alt } />
                <div className={ `${ stylesHome.text } ${ stylesHome.hideMobile }` }>
                  <h4>{ interest.brief }</h4>
                </div>
              </div>
            ) }
          </div>
          <div className={ stylesHome.openInterests }>
            <img src={ require('../images/' + openInterest.image + '.jpg' ) } alt={ openInterest.alt } style={{ width: "200px", height: "auto", objectFit: "contain", padding: "1em" }} />
            <div className={ stylesHome.text }>
              <h4>{ openInterest.brief }</h4>
              <div dangerouslySetInnerHTML={ { __html: openInterest.content } }/>
            </div>
          </div>
        </div>
      </div>
    )
  }
}