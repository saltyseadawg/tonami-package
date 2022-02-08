import React, { Component } from 'react';
import styles from '../SCSS/Main.module.scss';
import { Button, Row, Col, Modal, ModalHeader, ModalBody, ModalFooter } from'reactstrap';

export class ProjectSummary extends Component {
	render() {
		const source = this.props.source;
		return (
			<div className={ styles.projectSummary }>
				<img src={ require('../images/' + source.image.src + '.jpg' ) } alt={ source.image.alt } style={{objectFit: "contain", maxWidth: "400px", maxHeight: "500px" }} width="80%" height="80%" />
				<h1>{ source.projectName }</h1>
				<h3>{ source.tagline }</h3>
				<h4>{ source.type }</h4>
				{ source.link && 
					<a href={ source.link }><button className="btn btn-primary">Test out project!</button></a> }
				{ ( source.appStore || source.playStore ) && 
					<div className={ styles.badgeContainer }>
						{ source.appStore && 
							<a href={ source.appStore } target="_blank" rel="noopener noreferrer"><img alt='Download from the App Store' src={ require('../images/AppStoreBadge.svg' ) } height='65'/></a> }
						{ source.playStore && 
							<a href={ source.playStore } target="_blank" rel="noopener noreferrer"><img alt='Get it on Google Play' src={ require('../images/PlayStoreBadge.png' ) } height='65'/></a> }
					</div> }
				<div className={ styles.summary } dangerouslySetInnerHTML={ { __html: source.summary } } />
			</div>
		)
	}
}

export class AwardsComponent extends Component {
	render() {
		const source = this.props.source;
		return (
			<div className={ styles.awardsComponent }>
				<img src={ require('../images/' + source.image.src + '.jpg' ) } class="w3-round w3-image" alt={ source.image.alt } style={{objectFit: "contain", width:"15em", height:"15em" }} />
				<h4>{ source.header }</h4>
				{ source.caption && 
					<div style={{ textAlign: "center" }} dangerouslySetInnerHTML={ { __html: source.caption } } />
				}
			</div>
		)
	}
}

export class SlideshowComponent extends Component {
	constructor(props) {
    super(props);

    var modalStates = [];
    this.props.source.slideshow.forEach(function(element) {
      modalStates.push( false );
    });

    this.state={
      source: this.props.source,
      modalStates: modalStates,
      hover: undefined
    }
  }

  toggleModal( key ) {
    var modalStates = this.state.modalStates;
    modalStates[ key ] = !this.state.modalStates[ key ];
    this.setState({
      modalStates: modalStates
    });
  }

  render(){
  	const source = this.props.source;
    return(
	    <div style={{ marginTop: "3em", display: "block", overflow: "auto" }}>
		    <div className={ styles.slideshowComponent }>
          { source.slideshow.map( ( image, i ) =>
            <Col key={ i } onClick={ () => this.toggleModal( i ) } className={ styles.slidePicture } onMouseEnter={ () => this.setState({ hover: i })} onMouseLeave={ () => this.setState({ hover: undefined })} style={ ( this.state.hover !== undefined && i!== this.state.hover ) ? {opacity: 0.6} : {opacity: 1}}>
              <img src={ require('../images/' + image.image.src + '.jpg')} style={{ width: "100%", maxWidth: "300px", minWidth: "100px"}} alt={ image.image.alt }/>
              <Modal isOpen={ this.state.modalStates[ i ] } toggle={ () => this.toggleModal( i ) } size="lg">
                <ModalHeader toggle={ () => this.toggleModal( i ) }>{ image.title }</ModalHeader>
                <ModalBody>
                  <img src={ require('../images/' + image.image.src + '.jpg')} style={{ width: "100%" }} alt={ image.image.alt }/>
                  <div dangerouslySetInnerHTML={ { __html: image.description } } />
                </ModalBody>
                { source.slideshow.length !== 1 && 
	                <ModalFooter>
	                	{ i !== 0 && 
	                		<Button onClick={ () => { this.toggleModal( i ); this.toggleModal( i - 1) }}>Previous</Button>
	                	}
	                	{ i !== source.slideshow.length - 1 && 
	                		<Button onClick={ () => { this.toggleModal( i ); this.toggleModal( i + 1) }}>Next</Button>
	                	}
	                </ModalFooter>
	              }
              </Modal>
            </Col>
          ) }
        </div>
	    </div>
    )
  }
}

export class TableComponent extends Component {
	constructor(props) {
		super(props);
		this.state={
			up: true
		}
	}

  render(){
  	const source = this.props.source;
    return(
    <div style={{ marginTop: "3em", display: "block", overflow: "auto" }}>
	    <div className={ styles.tableComponent }>
	      <div className={ styles.main }>
	        <h2><div className={ styles.title }>
	          { source.title }
	        </div></h2>
	        { source.link && <a href={ source.link } target='_blank' rel='noopener noreferrer' >GitHub Repository</a> }
			<h4><Row>
			<Col md={ 2 } onClick={ () => this.setState({ up: !this.state.up })}>{ source.leftTitle } <i class={ this.state.up ? "fa fa-sort-up" : "fa fa-sort-down" } style={ this.state.up ? { transform: "translateY(25%)" } : { transform: "translateY(-25%)" } } /></Col>
			<Col md={ 10 } className={ styles.rightTitle }>{ source.rightTitle }</Col>
			</Row></h4>
			<div style={ this.state.up ? { display: "flex", flexDirection: "column" } : { display: "flex", flexDirection: "column-reverse" } }>
			{ source.list.map( ( item, i ) =>
				<Row key={ i } className={ `${ styles.listItem } listItem` }>
					<Col md={ 2 } className={ styles.left } dangerouslySetInnerHTML={ { __html: item.left } } />
					<Col md={ 10 } className={ styles.right }>
						{ item.right.map( ( point, j ) =>
							<div dangerouslySetInnerHTML={ { __html: point } } key={ j }/>
							) }
						</Col>
				</Row>
			) }
			</div>
	      </div>
	    </div>
    </div>
    )
  }
}

export class ListComponent extends Component {
  render(){
  	const source = this.props.source;
    return(
    <div style={{ marginTop: "3em", display: "block", overflow: "auto", overflowWrap: "break-word" }}>
	    <div className={ styles.listComponent }>
		    <div className={ styles.main }>
				<h2><div className={ styles.title }>
					{ source.title }
				</div></h2>
				<ul>
					{ source.list.map( ( point, i ) =>
						<li key={ i } dangerouslySetInnerHTML={ { __html: point } } />
					) }
				</ul>
		    </div>
	    </div>
    </div>
    )
  }
}