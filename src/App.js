import React, { Component } from 'react';
import './SCSS/App.scss';
import Home from './pages/HomePage';
import Exercise1 from './pages/Exercise1';
import Exercise2 from './pages/Exercise2';

export default class App extends Component {
  constructor() {
    super();

    this.state={
      page: 0
		}

    this.changePage = this.changePage.bind( this );
  }

  changePage(newPage) {
    this.setState({page: newPage});
  }

  render() {
    return (
      <div className="app-container" id="main">
        <div className="toHome" onClick={ () => this.changePage(0) }><i className="fas fa-home" /></div>
        {
          {
            0: <Home changePage={this.changePage}/>,
            1: <Exercise1 />,
            2: <Exercise2 />
          }[this.state.page]
        }
      </div>
    );
  }
}
